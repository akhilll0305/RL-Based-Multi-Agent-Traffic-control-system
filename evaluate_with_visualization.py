"""
Presentation-Ready Evaluation with Live Dashboard + Rich Console Logging + Post-Eval Diagrams

Demonstrates supervisor inter-zone communication visually during evaluation.

Usage:
  python evaluate_with_visualization.py                           # Fine-tuned model (default)
  python evaluate_with_visualization.py --model scratch           # From-scratch model
  python evaluate_with_visualization.py --model finetuned --gui   # With SUMO GUI
  python evaluate_with_visualization.py --episodes 3              # Custom episode count
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Interactive backend for live dashboard
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Dashboard disabled.")

from sumo_environment_federated import FederatedSumoEnvironment
from agent import DDQNAgent
from supervisor_agent import SupervisorAgent


# ==================== Color Constants ====================
class Colors:
    """ANSI color codes for rich console output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    BG_BLUE = '\033[44m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


# ==================== Action Labels ====================
SUP_ACTIONS = {0: 'NS_PRIORITY', 1: 'EW_PRIORITY', 2: 'BALANCED'}
SUP_COLORS = {0: Colors.GREEN, 1: Colors.YELLOW, 2: Colors.CYAN}
SUP_SYMBOLS = {0: '↕ NS', 1: '↔ EW', 2: '⊕ BAL'}


class CommunicationLogger:
    """Tracks all inter-zone communication events for visualization"""

    def __init__(self):
        self.events = []
        self.step_data = []
        self.supervisor_decisions = {'zone_a': [], 'zone_b': []}
        self.zone_metrics = {'zone_a': [], 'zone_b': []}
        self.cross_zone_flows = []
        self.reward_modifiers = []

    def log_supervisor_decision(self, step, zone, action, own_state, neighbor_state):
        """Log a supervisor coordination decision"""
        self.events.append({
            'step': step,
            'type': 'supervisor_decision',
            'zone': zone,
            'action': action,
            'action_name': SUP_ACTIONS[action],
            'own_queue': float(np.mean(own_state[:4])),
            'neighbor_queue': float(np.mean(neighbor_state[:4])),
            'cross_inflow': float(own_state[8]) if len(own_state) > 8 else 0,
            'cross_outflow': float(own_state[9]) if len(own_state) > 9 else 0,
        })
        self.supervisor_decisions[zone].append({
            'step': step, 'action': action
        })

    def log_zone_metrics(self, step, zone, queue, waiting_time, vehicles):
        """Log zone-level metrics"""
        self.zone_metrics[zone].append({
            'step': step, 'queue': queue, 'waiting_time': waiting_time, 'vehicles': vehicles
        })

    def log_cross_zone_flow(self, step, inflow_a, outflow_a, inflow_b, outflow_b):
        """Log cross-zone vehicle flow"""
        self.cross_zone_flows.append({
            'step': step,
            'a_to_b': outflow_a,
            'b_to_a': outflow_b,
        })

    def log_reward_modifier(self, step, tls_id, zone, modifier, sup_action):
        """Log reward modifier applied by supervisor"""
        self.reward_modifiers.append({
            'step': step, 'tls': tls_id, 'zone': zone,
            'modifier': modifier, 'sup_action': sup_action
        })


class LiveDashboard:
    """Real-time matplotlib dashboard showing supervisor communication"""

    def __init__(self, enabled=True):
        self.enabled = enabled and HAS_MATPLOTLIB
        self.fig = None
        self.axes = {}
        self.initialized = False

        # Data buffers
        self.steps = []
        self.zone_a_actions = []
        self.zone_b_actions = []
        self.zone_a_queue = []
        self.zone_b_queue = []
        self.cross_a_to_b = []
        self.cross_b_to_a = []
        self.zone_a_wait = []
        self.zone_b_wait = []

    def init_figure(self):
        """Create the dashboard figure"""
        if not self.enabled:
            return

        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('FEDERATED HIERARCHICAL TRAFFIC CONTROL — LIVE DASHBOARD',
                         fontsize=14, fontweight='bold')

        gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

        # Top-left: Network Architecture diagram
        self.axes['arch'] = self.fig.add_subplot(gs[0, 0])
        # Top-center: Current supervisor decisions
        self.axes['decisions'] = self.fig.add_subplot(gs[0, 1])
        # Top-right: Cross-zone flow
        self.axes['flow'] = self.fig.add_subplot(gs[0, 2])

        # Middle-left: Zone A queue over time
        self.axes['queue_a'] = self.fig.add_subplot(gs[1, 0])
        # Middle-center: Zone B queue over time
        self.axes['queue_b'] = self.fig.add_subplot(gs[1, 1])
        # Middle-right: Cross-zone traffic flow over time
        self.axes['cross_flow'] = self.fig.add_subplot(gs[1, 2])

        # Bottom: Supervisor action timeline (both zones)
        self.axes['timeline'] = self.fig.add_subplot(gs[2, :])

        self._draw_architecture()
        self.initialized = True
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_architecture(self):
        """Draw the static network architecture diagram"""
        ax = self.axes['arch']
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect('equal')
        ax.set_title('System Architecture', fontweight='bold', fontsize=10)
        ax.axis('off')

        # Zone A box
        zone_a_rect = mpatches.FancyBboxPatch((0, 0.5), 2.2, 3.0,
            boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
        ax.add_patch(zone_a_rect)
        ax.text(1.1, 3.7, 'ZONE A', ha='center', fontweight='bold', color='#1565C0', fontsize=9)

        # Zone B box
        zone_b_rect = mpatches.FancyBboxPatch((3.3, 0.5), 2.2, 3.0,
            boxstyle="round,pad=0.1", facecolor='#FBE9E7', edgecolor='#D84315', linewidth=2)
        ax.add_patch(zone_b_rect)
        ax.text(4.4, 3.7, 'ZONE B', ha='center', fontweight='bold', color='#D84315', fontsize=9)

        # TLS nodes - Zone A
        for i, (x, y) in enumerate([(0.5, 2.8), (1.7, 2.8), (0.5, 1.2), (1.7, 1.2)]):
            circle = plt.Circle((x, y), 0.25, color='#2196F3', zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, f'T{i+1}', ha='center', va='center', color='white',
                   fontweight='bold', fontsize=7, zorder=6)

        # TLS nodes - Zone B
        for i, (x, y) in enumerate([(3.8, 2.8), (5.0, 2.8), (3.8, 1.2), (5.0, 1.2)]):
            circle = plt.Circle((x, y), 0.25, color='#FF5722', zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, f'T{i+5}', ha='center', va='center', color='white',
                   fontweight='bold', fontsize=7, zorder=6)

        # Supervisor nodes
        sup_a = plt.Circle((1.1, 0.1), 0.3, color='#1565C0', zorder=5)
        ax.add_patch(sup_a)
        ax.text(1.1, 0.1, 'SUP\nA', ha='center', va='center', color='white',
               fontweight='bold', fontsize=6, zorder=6)

        sup_b = plt.Circle((4.4, 0.1), 0.3, color='#D84315', zorder=5)
        ax.add_patch(sup_b)
        ax.text(4.4, 0.1, 'SUP\nB', ha='center', va='center', color='white',
               fontweight='bold', fontsize=6, zorder=6)

        # Inter-zone communication arrow (bidirectional)
        ax.annotate('', xy=(4.0, 0.1), xytext=(1.5, 0.1),
                   arrowprops=dict(arrowstyle='<->', color='#E65100', lw=2.5))
        ax.text(2.75, -0.2, 'INTER-ZONE\nCOMMUNICATION', ha='center', fontsize=6,
               color='#E65100', fontweight='bold')

        # Cross-zone bridge arrows
        ax.annotate('', xy=(3.55, 2.8), xytext=(1.95, 2.8),
                   arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=1.5, linestyle='dashed'))
        ax.annotate('', xy=(3.55, 1.2), xytext=(1.95, 1.2),
                   arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=1.5, linestyle='dashed'))
        ax.text(2.75, 3.1, 'bridge', ha='center', fontsize=6, color='#FF9800', style='italic')
        ax.text(2.75, 1.5, 'bridge', ha='center', fontsize=6, color='#FF9800', style='italic')

    def update(self, step, zone_a_state, zone_b_state, sup_a_action, sup_b_action,
               cross_a_to_b, cross_b_to_a, zone_a_q, zone_b_q, zone_a_w, zone_b_w):
        """Update all dashboard panels with new data"""
        if not self.enabled or not self.initialized:
            return

        self.steps.append(step)
        self.zone_a_actions.append(sup_a_action)
        self.zone_b_actions.append(sup_b_action)
        self.zone_a_queue.append(zone_a_q)
        self.zone_b_queue.append(zone_b_q)
        self.cross_a_to_b.append(cross_a_to_b)
        self.cross_b_to_a.append(cross_b_to_a)
        self.zone_a_wait.append(zone_a_w)
        self.zone_b_wait.append(zone_b_w)

        try:
            self._update_decisions(sup_a_action, sup_b_action, zone_a_q, zone_b_q)
            self._update_flow(cross_a_to_b, cross_b_to_a)
            self._update_queue_plots()
            self._update_cross_flow_plot()
            self._update_timeline()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception:
            pass

    def _update_decisions(self, sup_a_action, sup_b_action, za_q, zb_q):
        """Update current supervisor decisions panel"""
        ax = self.axes['decisions']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Current Supervisor Decisions', fontweight='bold', fontsize=10)

        # Zone A decision
        a_color = ['#4CAF50', '#FFC107', '#2196F3'][sup_a_action]
        a_rect = mpatches.FancyBboxPatch((0.5, 5.5), 3.5, 3.5,
            boxstyle="round,pad=0.2", facecolor=a_color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(a_rect)
        ax.text(2.25, 8.2, 'SUP A', ha='center', fontweight='bold', color='white', fontsize=10)
        ax.text(2.25, 6.8, SUP_ACTIONS[sup_a_action], ha='center', color='white', fontsize=9)
        ax.text(2.25, 5.8, f'Queue: {za_q:.1f}', ha='center', color='white', fontsize=8)

        # Zone B decision
        b_color = ['#4CAF50', '#FFC107', '#2196F3'][sup_b_action]
        b_rect = mpatches.FancyBboxPatch((5.5, 5.5), 3.5, 3.5,
            boxstyle="round,pad=0.2", facecolor=b_color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(b_rect)
        ax.text(7.25, 8.2, 'SUP B', ha='center', fontweight='bold', color='white', fontsize=10)
        ax.text(7.25, 6.8, SUP_ACTIONS[sup_b_action], ha='center', color='white', fontsize=9)
        ax.text(7.25, 5.8, f'Queue: {zb_q:.1f}', ha='center', color='white', fontsize=8)

        # Communication arrow
        ax.annotate('', xy=(5.3, 7.2), xytext=(4.2, 7.2),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
        ax.text(4.75, 6.3, 'SHARING\nZONE STATE', ha='center', fontsize=7,
               color='red', fontweight='bold')

        # Legend
        for i, (name, color) in enumerate([('NS Priority', '#4CAF50'),
                                            ('EW Priority', '#FFC107'),
                                            ('Balanced', '#2196F3')]):
            ax.add_patch(mpatches.Rectangle((1 + i * 3, 0.5), 2.5, 1.2,
                        facecolor=color, alpha=0.6))
            ax.text(2.25 + i * 3, 1.1, name, ha='center', fontsize=7, fontweight='bold')

    def _update_flow(self, a_to_b, b_to_a):
        """Update cross-zone flow visualization"""
        ax = self.axes['flow']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Cross-Zone Vehicle Flow', fontweight='bold', fontsize=10)

        # Zone boxes
        ax.add_patch(mpatches.FancyBboxPatch((0.5, 3), 3, 4,
            boxstyle="round,pad=0.2", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2))
        ax.text(2, 5.5, 'ZONE A', ha='center', fontweight='bold', color='#1565C0', fontsize=12)
        ax.text(2, 4.2, f'{a_to_b:.0f} veh →', ha='center', fontsize=10, color='#E65100')

        ax.add_patch(mpatches.FancyBboxPatch((6.5, 3), 3, 4,
            boxstyle="round,pad=0.2", facecolor='#FBE9E7', edgecolor='#D84315', linewidth=2))
        ax.text(8, 5.5, 'ZONE B', ha='center', fontweight='bold', color='#D84315', fontsize=12)
        ax.text(8, 4.2, f'← {b_to_a:.0f} veh', ha='center', fontsize=10, color='#E65100')

        # Flow arrows with thickness proportional to flow
        a_lw = max(1, min(a_to_b, 10))
        b_lw = max(1, min(b_to_a, 10))
        ax.annotate('', xy=(6.3, 5.8), xytext=(3.7, 5.8),
                   arrowprops=dict(arrowstyle='->', color='#E65100', lw=a_lw))
        ax.annotate('', xy=(3.7, 4.8), xytext=(6.3, 4.8),
                   arrowprops=dict(arrowstyle='->', color='#E65100', lw=b_lw))

        ax.text(5, 2, f'Net Flow: {abs(a_to_b - b_to_a):.0f} veh '
               f'{"A→B" if a_to_b > b_to_a else "B→A"}',
               ha='center', fontsize=9, fontweight='bold')

    def _update_queue_plots(self):
        """Update zone queue time-series"""
        for zone, key, data, color, label in [
            ('queue_a', 'zone_a', self.zone_a_queue, '#1565C0', 'Zone A'),
            ('queue_b', 'zone_b', self.zone_b_queue, '#D84315', 'Zone B')
        ]:
            ax = self.axes[zone]
            ax.clear()
            ax.plot(self.steps, data, color=color, linewidth=1.5)
            ax.fill_between(self.steps, data, alpha=0.2, color=color)
            ax.set_xlabel('Step', fontsize=8)
            ax.set_ylabel('Avg Queue', fontsize=8)
            ax.set_title(f'{label} Queue', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3)

    def _update_cross_flow_plot(self):
        """Update cross-zone flow time-series"""
        ax = self.axes['cross_flow']
        ax.clear()
        ax.plot(self.steps, self.cross_a_to_b, color='#1565C0', linewidth=1.2, label='A → B')
        ax.plot(self.steps, self.cross_b_to_a, color='#D84315', linewidth=1.2, label='B → A')
        ax.fill_between(self.steps, self.cross_a_to_b, alpha=0.15, color='#1565C0')
        ax.fill_between(self.steps, self.cross_b_to_a, alpha=0.15, color='#D84315')
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Vehicles', fontsize=8)
        ax.set_title('Cross-Zone Traffic', fontweight='bold', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _update_timeline(self):
        """Update supervisor action timeline"""
        ax = self.axes['timeline']
        ax.clear()

        colors_map = {0: '#4CAF50', 1: '#FFC107', 2: '#2196F3'}

        if self.steps:
            # Zone A actions as colored bars
            for i, (s, a) in enumerate(zip(self.steps, self.zone_a_actions)):
                width = self.steps[1] - self.steps[0] if len(self.steps) > 1 else 1
                ax.barh(1.5, width, left=s - width/2, height=0.6,
                       color=colors_map[a], edgecolor='none', alpha=0.8)

            # Zone B actions
            for i, (s, a) in enumerate(zip(self.steps, self.zone_b_actions)):
                width = self.steps[1] - self.steps[0] if len(self.steps) > 1 else 1
                ax.barh(0.5, width, left=s - width/2, height=0.6,
                       color=colors_map[a], edgecolor='none', alpha=0.8)

        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Supervisor B', 'Supervisor A'], fontweight='bold')
        ax.set_xlabel('Step', fontsize=9)
        ax.set_title('Supervisor Action Timeline (showing inter-zone coordination)',
                    fontweight='bold', fontsize=10)
        ax.set_ylim(-0.1, 2.3)

        # Legend
        patches = [mpatches.Patch(color='#4CAF50', label='NS Priority'),
                   mpatches.Patch(color='#FFC107', label='EW Priority'),
                   mpatches.Patch(color='#2196F3', label='Balanced')]
        ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3, axis='x')

    def reset_episode(self):
        """Clear data for new episode"""
        self.steps.clear()
        self.zone_a_actions.clear()
        self.zone_b_actions.clear()
        self.zone_a_queue.clear()
        self.zone_b_queue.clear()
        self.cross_a_to_b.clear()
        self.cross_b_to_a.clear()
        self.zone_a_wait.clear()
        self.zone_b_wait.clear()

    def close(self):
        if self.enabled and self.fig:
            plt.ioff()
            plt.close(self.fig)


def print_header(text, color=Colors.CYAN):
    """Print a styled header"""
    width = 70
    print(f"\n{color}{Colors.BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{Colors.RESET}")


def print_communication_event(step, zone, action, own_q, neighbor_q, cross_in, cross_out):
    """Print a rich communication event to console"""
    color = SUP_COLORS[action]
    symbol = SUP_SYMBOLS[action]
    zone_color = Colors.BLUE if zone == 'zone_a' else Colors.RED
    zone_label = 'ZONE A' if zone == 'zone_a' else 'ZONE B'
    neighbor = 'ZONE B' if zone == 'zone_a' else 'ZONE A'

    print(f"  {Colors.DIM}Step {step:>4}{Colors.RESET}  "
          f"{zone_color}{Colors.BOLD}{zone_label}{Colors.RESET} SUP → "
          f"{color}{Colors.BOLD}{symbol}{Colors.RESET}  "
          f"│ Own Q: {own_q:>5.1f} │ {neighbor} Q: {neighbor_q:>5.1f} │ "
          f"Cross: ↓{cross_in:.0f} ↑{cross_out:.0f}")


def evaluate_with_visualization(checkpoint_dir, results_dir, label,
                                 use_gui=False, num_episodes=3):
    """
    Run evaluation with full visualization suite.
    """
    print_header(f'EVALUATING: {label}')
    print(f"  {Colors.DIM}Checkpoints: {checkpoint_dir}{Colors.RESET}")
    print(f"  {Colors.DIM}Episodes:    {num_episodes}{Colors.RESET}")

    # Initialize environment
    env = FederatedSumoEnvironment(use_gui=use_gui)
    local_state_dim = env.get_local_state_dim()
    action_dim = env.get_action_dim()
    zone_state_dim = env.get_zone_state_dim()

    # Load agents
    local_agents = {}
    for tls_id in env.tls_ids:
        agent = DDQNAgent(state_dim=local_state_dim, action_dim=action_dim)
        ckpt = f'{checkpoint_dir}/{tls_id}_episode_final.pth'
        if os.path.exists(ckpt):
            agent.load(ckpt)
            print(f"  {Colors.GREEN}✓{Colors.RESET} Loaded {tls_id}")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} No checkpoint: {tls_id}")
        local_agents[tls_id] = agent

    supervisor_a = SupervisorAgent('zone_a', state_dim=zone_state_dim * 2, action_dim=3)
    supervisor_b = SupervisorAgent('zone_b', state_dim=zone_state_dim * 2, action_dim=3)

    for sup, name, ckpt_name in [
        (supervisor_a, 'Supervisor A', 'supervisor_a_episode_final.pth'),
        (supervisor_b, 'Supervisor B', 'supervisor_b_episode_final.pth')
    ]:
        ckpt = f'{checkpoint_dir}/{ckpt_name}'
        if os.path.exists(ckpt):
            sup.load(ckpt)
            print(f"  {Colors.GREEN}✓{Colors.RESET} Loaded {name}")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} No checkpoint: {name}")

    # Initialize dashboard
    dashboard = LiveDashboard(enabled=HAS_MATPLOTLIB)
    dashboard.init_figure()

    logger = CommunicationLogger()
    all_metrics = []

    for ep in range(num_episodes):
        print_header(f'EPISODE {ep + 1}/{num_episodes}', Colors.MAGENTA)
        dashboard.reset_episode()

        states = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        # Initial supervisor decisions
        zone_a_state = env.get_zone_state('zone_a')
        zone_b_state = env.get_zone_state('zone_b')
        sup_a_action = supervisor_a.select_action(zone_a_state, zone_b_state, training=False)
        sup_b_action = supervisor_b.select_action(zone_b_state, zone_a_state, training=False)

        print(f"\n  {Colors.BOLD}{'─' * 66}")
        print(f"  {'Step':>6}  {'ZONE A SUP':>12}  {'ZONE B SUP':>12}  "
              f"{'A Queue':>8}  {'B Queue':>8}  {'Cross':>8}")
        print(f"  {'─' * 66}{Colors.RESET}")

        prev_a_action = -1
        prev_b_action = -1

        while not done:
            step_count += 1

            # Supervisor decisions every 3 steps
            if step_count % 3 == 0:
                zone_a_state = env.get_zone_state('zone_a')
                zone_b_state = env.get_zone_state('zone_b')
                sup_a_action = supervisor_a.select_action(zone_a_state, zone_b_state, training=False)
                sup_b_action = supervisor_b.select_action(zone_b_state, zone_a_state, training=False)

                # Calculate metrics for logging
                za_q = float(np.mean(zone_a_state[:4]))
                zb_q = float(np.mean(zone_b_state[:4]))
                za_wait = float(zone_a_state[5]) if len(zone_a_state) > 5 else 0
                zb_wait = float(zone_b_state[5]) if len(zone_b_state) > 5 else 0
                cross_a_in = float(zone_a_state[8]) if len(zone_a_state) > 8 else 0
                cross_a_out = float(zone_a_state[9]) if len(zone_a_state) > 9 else 0
                cross_b_in = float(zone_b_state[8]) if len(zone_b_state) > 8 else 0
                cross_b_out = float(zone_b_state[9]) if len(zone_b_state) > 9 else 0

                # Log events
                logger.log_supervisor_decision(step_count, 'zone_a', sup_a_action,
                                               zone_a_state, zone_b_state)
                logger.log_supervisor_decision(step_count, 'zone_b', sup_b_action,
                                               zone_b_state, zone_a_state)
                logger.log_cross_zone_flow(step_count, cross_a_in, cross_a_out,
                                           cross_b_in, cross_b_out)

                # Rich console output - highlight action changes
                a_changed = sup_a_action != prev_a_action
                b_changed = sup_b_action != prev_b_action
                a_marker = f"{Colors.BOLD}*" if a_changed else " "
                b_marker = f"{Colors.BOLD}*" if b_changed else " "

                a_color = SUP_COLORS[sup_a_action]
                b_color = SUP_COLORS[sup_b_action]

                print(f"  {Colors.DIM}{step_count:>6}{Colors.RESET}  "
                      f"{a_color}{a_marker}{SUP_SYMBOLS[sup_a_action]:>10}{Colors.RESET}  "
                      f"{b_color}{b_marker}{SUP_SYMBOLS[sup_b_action]:>10}{Colors.RESET}  "
                      f"{za_q:>8.1f}  {zb_q:>8.1f}  "
                      f"{Colors.DIM}A→B:{cross_a_out:.0f} B→A:{cross_b_out:.0f}{Colors.RESET}")

                if a_changed or b_changed:
                    changed_zone = []
                    if a_changed:
                        changed_zone.append(f"Zone A: {SUP_ACTIONS.get(prev_a_action, '?')} → {SUP_ACTIONS[sup_a_action]}")
                    if b_changed:
                        changed_zone.append(f"Zone B: {SUP_ACTIONS.get(prev_b_action, '?')} → {SUP_ACTIONS[sup_b_action]}")
                    print(f"  {Colors.YELLOW}  ⟳ STRATEGY CHANGE: {', '.join(changed_zone)}{Colors.RESET}")

                prev_a_action = sup_a_action
                prev_b_action = sup_b_action

                # Update dashboard
                dashboard.update(step_count, zone_a_state, zone_b_state,
                               sup_a_action, sup_b_action,
                               cross_a_out, cross_b_out, za_q, zb_q, za_wait, zb_wait)

            # Local agents act greedily
            actions = {}
            for tls_id in env.tls_ids:
                actions[tls_id] = local_agents[tls_id].select_action(states[tls_id], training=False)

            next_states, rewards, done, info = env.step(actions)
            total_reward += sum(rewards.values())
            states = next_states

        # Episode summary
        metrics = {
            'episode': ep + 1,
            'total_reward': total_reward,
            'avg_waiting_time': info['avg_waiting_time'],
            'total_vehicles': info['total_vehicles'],
            'zone_a_queue': info['per_zone']['zone_a']['total_queue'],
            'zone_b_queue': info['per_zone']['zone_b']['total_queue'],
        }
        all_metrics.append(metrics)

        print(f"\n  {Colors.BOLD}Episode {ep + 1} Results:{Colors.RESET}")
        print(f"    {Colors.GREEN}Reward:{Colors.RESET}       {total_reward:.1f}")
        print(f"    {Colors.GREEN}Avg Wait:{Colors.RESET}     {info['avg_waiting_time']:.1f}s")
        print(f"    {Colors.GREEN}Zone A Queue:{Colors.RESET} {info['per_zone']['zone_a']['total_queue']}")
        print(f"    {Colors.GREEN}Zone B Queue:{Colors.RESET} {info['per_zone']['zone_b']['total_queue']}")
        print(f"    {Colors.GREEN}Vehicles:{Colors.RESET}     {info['total_vehicles']}")

    env.close()
    dashboard.close()

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    eval_path = f'{results_dir}/evaluation_results.csv'
    with open(eval_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)

    return all_metrics, logger


def generate_communication_diagram(logger, output_dir='results/communication'):
    """
    Generate post-evaluation communication diagrams.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot generate diagrams without matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ==================== Diagram 1: Communication Timeline ====================
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 2, 1.5]})
    fig.suptitle('Supervisor Communication & Coordination Timeline', fontsize=14, fontweight='bold')

    colors_map = {0: '#4CAF50', 1: '#FFC107', 2: '#2196F3'}

    # Panel 1: Supervisor decisions over time
    ax = axes[0]
    a_steps = [e['step'] for e in logger.supervisor_decisions['zone_a']]
    a_acts = [e['action'] for e in logger.supervisor_decisions['zone_a']]
    b_steps = [e['step'] for e in logger.supervisor_decisions['zone_b']]
    b_acts = [e['action'] for e in logger.supervisor_decisions['zone_b']]

    if a_steps:
        for i in range(len(a_steps) - 1):
            ax.barh(1, a_steps[i+1] - a_steps[i], left=a_steps[i], height=0.4,
                   color=colors_map[a_acts[i]], edgecolor='none')
        for i in range(len(b_steps) - 1):
            ax.barh(0, b_steps[i+1] - b_steps[i], left=b_steps[i], height=0.4,
                   color=colors_map[b_acts[i]], edgecolor='none')

    # Mark action changes with vertical lines
    for i in range(1, len(a_acts)):
        if a_acts[i] != a_acts[i-1]:
            ax.axvline(a_steps[i], color='black', linewidth=0.8, alpha=0.5, linestyle='--')
    for i in range(1, len(b_acts)):
        if b_acts[i] != b_acts[i-1]:
            ax.axvline(b_steps[i], color='black', linewidth=0.8, alpha=0.5, linestyle='--')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Supervisor B\n(Zone B)', 'Supervisor A\n(Zone A)'])
    ax.set_title('Supervisor Coordination Decisions', fontweight='bold')
    patches = [mpatches.Patch(color='#4CAF50', label='NS Priority'),
               mpatches.Patch(color='#FFC107', label='EW Priority'),
               mpatches.Patch(color='#2196F3', label='Balanced')]
    ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='x')

    # Panel 2: Cross-zone flows + queue levels
    ax = axes[1]
    flow_steps = [e['step'] for e in logger.cross_zone_flows]
    a_to_b = [e['a_to_b'] for e in logger.cross_zone_flows]
    b_to_a = [e['b_to_a'] for e in logger.cross_zone_flows]

    if flow_steps:
        ax.fill_between(flow_steps, a_to_b, alpha=0.3, color='#1565C0', label='A → B flow')
        ax.fill_between(flow_steps, b_to_a, alpha=0.3, color='#D84315', label='B → A flow')
        ax.plot(flow_steps, a_to_b, color='#1565C0', linewidth=1.5)
        ax.plot(flow_steps, b_to_a, color='#D84315', linewidth=1.5)

    ax.set_ylabel('Cross-Zone Vehicles')
    ax.set_title('Inter-Zone Vehicle Flow (communication triggers)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Coordination analysis
    ax = axes[2]
    # Show when both supervisors agree vs disagree
    if a_steps and b_steps:
        min_len = min(len(a_acts), len(b_acts))
        agreement = [1 if a_acts[i] == b_acts[i] else 0 for i in range(min_len)]
        steps_common = a_steps[:min_len]

        ax.bar(steps_common, agreement, width=3, color='#4CAF50', alpha=0.6, label='Agree')
        ax.bar(steps_common, [1 - a for a in agreement], bottom=agreement,
              width=3, color='#F44336', alpha=0.6, label='Different Strategy')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Agreement')
    ax.set_title('Supervisor Strategy Agreement', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Different', 'Same'])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/communication_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Communication timeline saved to {output_dir}/communication_timeline.png")

    # ==================== Diagram 2: Architecture + Communication Flow ====================
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_title('Federated Hierarchical Architecture — Communication Flow',
                fontsize=14, fontweight='bold')

    # FedAvg Server (top)
    ax.add_patch(mpatches.FancyBboxPatch((5.5, 9), 3, 1.5,
        boxstyle="round,pad=0.2", facecolor='#9C27B0', edgecolor='black', linewidth=2, alpha=0.8))
    ax.text(7, 9.8, 'FedAvg Server', ha='center', fontweight='bold', color='white', fontsize=11)
    ax.text(7, 9.3, 'Weight Aggregation', ha='center', color='white', fontsize=8)

    # Zone A Supervisor
    ax.add_patch(mpatches.FancyBboxPatch((1, 6), 4, 2,
        boxstyle="round,pad=0.2", facecolor='#1565C0', edgecolor='black', linewidth=2, alpha=0.85))
    ax.text(3, 7.3, 'SUPERVISOR A', ha='center', fontweight='bold', color='white', fontsize=11)
    ax.text(3, 6.7, 'Zone-level coordination', ha='center', color='#BBDEFB', fontsize=8)
    ax.text(3, 6.2, '24-dim combined state → 3 actions', ha='center', color='#BBDEFB', fontsize=7)

    # Zone B Supervisor
    ax.add_patch(mpatches.FancyBboxPatch((9, 6), 4, 2,
        boxstyle="round,pad=0.2", facecolor='#D84315', edgecolor='black', linewidth=2, alpha=0.85))
    ax.text(11, 7.3, 'SUPERVISOR B', ha='center', fontweight='bold', color='white', fontsize=11)
    ax.text(11, 6.7, 'Zone-level coordination', ha='center', color='#FFCCBC', fontsize=8)
    ax.text(11, 6.2, '24-dim combined state → 3 actions', ha='center', color='#FFCCBC', fontsize=7)

    # Inter-supervisor communication
    ax.annotate('', xy=(8.8, 7), xytext=(5.2, 7),
               arrowprops=dict(arrowstyle='<->', color='#E65100', lw=3))
    ax.text(7, 7.5, 'ZONE STATE\nEXCHANGE', ha='center', fontsize=8,
           color='#E65100', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100'))

    # FedAvg arrows
    ax.annotate('', xy=(5.5, 9), xytext=(3, 8.2),
               arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=2, linestyle='dashed'))
    ax.annotate('', xy=(8.5, 9), xytext=(11, 8.2),
               arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=2, linestyle='dashed'))
    ax.text(3.5, 8.8, 'Inter-zone\nFedAvg', ha='center', fontsize=7, color='#7B1FA2')
    ax.text(10.5, 8.8, 'Inter-zone\nFedAvg', ha='center', fontsize=7, color='#7B1FA2')

    # Local Agents - Zone A
    for i, (x, tls) in enumerate([(0.5, 'TLS 1'), (2, 'TLS 2'), (3.5, 'TLS 3'), (5, 'TLS 4')]):
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.5, 2.5), 1.2, 1.5,
            boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5))
        ax.text(x + 0.1, 3.5, tls, ha='center', fontweight='bold', fontsize=8, color='#1565C0')
        ax.text(x + 0.1, 2.9, 'DDQN', ha='center', fontsize=7, color='#1565C0')
        # Arrow to supervisor
        ax.annotate('', xy=(3, 5.8), xytext=(x + 0.1, 4.1),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1, alpha=0.6))

    # Local Agents - Zone B
    for i, (x, tls) in enumerate([(9, 'TLS 5'), (10.5, 'TLS 6'), (12, 'TLS 7'), (13.5, 'TLS 8')]):
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.5, 2.5), 1.2, 1.5,
            boxstyle="round,pad=0.1", facecolor='#FBE9E7', edgecolor='#D84315', linewidth=1.5))
        ax.text(x + 0.1, 3.5, tls, ha='center', fontweight='bold', fontsize=8, color='#D84315')
        ax.text(x + 0.1, 2.9, 'DDQN', ha='center', fontsize=7, color='#D84315')
        ax.annotate('', xy=(11, 5.8), xytext=(x + 0.1, 4.1),
                   arrowprops=dict(arrowstyle='->', color='#D84315', lw=1, alpha=0.6))

    # Intra-zone FedAvg label
    ax.text(2.75, 4.7, 'Intra-zone\nFedAvg', ha='center', fontsize=7, color='#1565C0',
           fontweight='bold', style='italic')
    ax.text(11.25, 4.7, 'Intra-zone\nFedAvg', ha='center', fontsize=7, color='#D84315',
           fontweight='bold', style='italic')

    # Cross-zone bridges
    ax.annotate('', xy=(8.5, 3.2), xytext=(5.7, 3.2),
               arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=2, linestyle='dashed'))
    ax.text(7.1, 2.2, 'Cross-Zone\nBridge Traffic', ha='center', fontsize=8,
           color='#FF9800', fontweight='bold')

    # SUMO environment box
    ax.add_patch(mpatches.FancyBboxPatch((-0.5, 0.5), 15, 1.2,
        boxstyle="round,pad=0.1", facecolor='#F5F5F5', edgecolor='#616161', linewidth=1.5))
    ax.text(7, 1.1, 'SUMO Traffic Simulator — 8-Intersection 4×2 Grid Network',
           ha='center', fontsize=10, color='#424242', fontweight='bold')

    plt.savefig(f'{output_dir}/architecture_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Architecture diagram saved to {output_dir}/architecture_diagram.png")

    # ==================== Diagram 3: Communication Stats ====================
    events = logger.events
    if not events:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Supervisor Communication Statistics', fontsize=13, fontweight='bold')

    # Action distribution comparison
    ax = axes[0]
    a_actions = [e['action'] for e in events if e['zone'] == 'zone_a']
    b_actions = [e['action'] for e in events if e['zone'] == 'zone_b']
    labels = ['NS Priority', 'EW Priority', 'Balanced']
    x = np.arange(3)
    if a_actions and b_actions:
        a_counts = [a_actions.count(i) / len(a_actions) for i in range(3)]
        b_counts = [b_actions.count(i) / len(b_actions) for i in range(3)]
        ax.bar(x - 0.18, a_counts, 0.35, color='#1565C0', label='Sup A', alpha=0.8)
        ax.bar(x + 0.18, b_counts, 0.35, color='#D84315', label='Sup B', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fraction')
    ax.set_title('Action Distribution')
    ax.legend()

    # Response to cross-zone pressure
    ax = axes[1]
    a_queues = [e['own_queue'] for e in events if e['zone'] == 'zone_a']
    b_queues = [e['own_queue'] for e in events if e['zone'] == 'zone_b']
    a_neighbor_q = [e['neighbor_queue'] for e in events if e['zone'] == 'zone_a']
    b_neighbor_q = [e['neighbor_queue'] for e in events if e['zone'] == 'zone_b']
    if a_queues:
        steps_a = range(len(a_queues))
        ax.plot(steps_a, a_queues, color='#1565C0', linewidth=1, label='Zone A own Q')
        ax.plot(steps_a, a_neighbor_q, color='#1565C0', linewidth=1, linestyle='--', label='Zone A sees B Q')
        ax.plot(range(len(b_queues)), b_queues, color='#D84315', linewidth=1, label='Zone B own Q')
        ax.plot(range(len(b_neighbor_q)), b_neighbor_q, color='#D84315', linewidth=1, linestyle='--', label='Zone B sees A Q')
    ax.set_ylabel('Queue Length')
    ax.set_title('Cross-Zone Awareness')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Communication frequency
    ax = axes[2]
    total_decisions = len(a_actions) + len(b_actions)
    a_changes = sum(1 for i in range(1, len(a_actions)) if a_actions[i] != a_actions[i-1])
    b_changes = sum(1 for i in range(1, len(b_actions)) if b_actions[i] != b_actions[i-1])
    bars = ax.bar(['Sup A\nDecisions', 'Sup B\nDecisions', 'Sup A\nChanges', 'Sup B\nChanges'],
                  [len(a_actions), len(b_actions), a_changes, b_changes],
                  color=['#1565C0', '#D84315', '#1565C0', '#D84315'],
                  alpha=[0.8, 0.8, 0.5, 0.5])
    ax.set_title('Decision & Strategy Changes')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, [len(a_actions), len(b_actions), a_changes, b_changes]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/communication_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Communication stats saved to {output_dir}/communication_stats.png")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Federated Model with Communication Visualization'
    )
    parser.add_argument('--model', type=str, default='finetuned',
                        choices=['scratch', 'finetuned'],
                        help='Which model to evaluate (default: finetuned)')
    parser.add_argument('--gui', action='store_true',
                        help='Show SUMO GUI')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of evaluation episodes (default: 3)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable live matplotlib dashboard')

    args = parser.parse_args()

    if args.model == 'scratch':
        checkpoint_dir = 'checkpoints/federated'
        results_dir = 'results/federated'
        label = 'From Scratch (700 episodes)'
    else:
        checkpoint_dir = 'checkpoints/federated_finetuned'
        results_dir = 'results/federated_finetuned'
        label = 'Fine-Tuned (200 episodes)'

    if args.no_dashboard:
        global HAS_MATPLOTLIB
        HAS_MATPLOTLIB = False

    metrics, logger = evaluate_with_visualization(
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        label=label,
        use_gui=args.gui,
        num_episodes=args.episodes
    )

    # Generate post-evaluation diagrams
    print_header('GENERATING COMMUNICATION DIAGRAMS')
    generate_communication_diagram(logger, output_dir='results/communication')

    # Final summary
    print_header('EVALUATION COMPLETE', Colors.GREEN)
    avg_reward = np.mean([m['total_reward'] for m in metrics])
    avg_wait = np.mean([m['avg_waiting_time'] for m in metrics])
    print(f"  Model:      {label}")
    print(f"  Avg Reward: {avg_reward:.1f}")
    print(f"  Avg Wait:   {avg_wait:.1f}s")
    print(f"  Diagrams:   results/communication/")


if __name__ == '__main__':
    main()
