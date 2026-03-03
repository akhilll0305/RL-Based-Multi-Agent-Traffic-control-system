"""
Generate SUMO Network for Federated Hierarchical Traffic Control (4x2 Grid)
8 Intersections split into 2 Zones (Zone A: TLS 1-4, Zone B: TLS 5-8)

Layout:
    Zone A                    Zone B
  [TLS1] ── [TLS2] ────── [TLS5] ── [TLS6]
    |          |              |          |
  [TLS3] ── [TLS4] ────── [TLS7] ── [TLS8]

Each zone is a 2x2 grid. Zone A and Zone B are connected horizontally.
Supervisor A manages TLS 1-4, Supervisor B manages TLS 5-8.
"""

import os
import subprocess


# Grid layout coordinates (4 columns x 2 rows)
# Zone A: columns 0,1  |  Zone B: columns 2,3
GRID = {
    # Zone A
    'intersection_1': (-750, 250),   # Row 0, Col 0 (Top-Left)
    'intersection_2': (-250, 250),   # Row 0, Col 1
    'intersection_3': (-750, -250),  # Row 1, Col 0 (Bottom-Left)
    'intersection_4': (-250, -250),  # Row 1, Col 1
    # Zone B
    'intersection_5': (250, 250),    # Row 0, Col 2
    'intersection_6': (750, 250),    # Row 0, Col 3 (Top-Right)
    'intersection_7': (250, -250),   # Row 1, Col 2
    'intersection_8': (750, -250),   # Row 1, Col 3 (Bottom-Right)
}

TLS_IDS = [f'tls_{i}' for i in range(1, 9)]
ZONE_A = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
ZONE_B = ['tls_5', 'tls_6', 'tls_7', 'tls_8']

OUTPUT_DIR = 'sumo_config/federated'


def generate_nodes():
    """Generate node definitions for 4x2 grid network"""
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">

    <!-- ============ External Boundary Nodes (traffic sources/sinks) ============ -->
    <!-- North boundary (4 entry points, one per column) -->
    <node id="north_ext_1" x="-750" y="750" type="priority"/>
    <node id="north_ext_2" x="-250" y="750" type="priority"/>
    <node id="north_ext_3" x="250"  y="750" type="priority"/>
    <node id="north_ext_4" x="750"  y="750" type="priority"/>

    <!-- South boundary -->
    <node id="south_ext_1" x="-750" y="-750" type="priority"/>
    <node id="south_ext_2" x="-250" y="-750" type="priority"/>
    <node id="south_ext_3" x="250"  y="-750" type="priority"/>
    <node id="south_ext_4" x="750"  y="-750" type="priority"/>

    <!-- West boundary (2 entry points, one per row) -->
    <node id="west_ext_1" x="-1250" y="250"  type="priority"/>
    <node id="west_ext_2" x="-1250" y="-250" type="priority"/>

    <!-- East boundary -->
    <node id="east_ext_1" x="1250" y="250"  type="priority"/>
    <node id="east_ext_2" x="1250" y="-250" type="priority"/>

    <!-- ============ Intersection Nodes (traffic light controlled) ============ -->
    <!-- Zone A (left 2x2 grid) -->
    <node id="intersection_1" x="-750" y="250"  type="traffic_light" tl="tls_1"/>
    <node id="intersection_2" x="-250" y="250"  type="traffic_light" tl="tls_2"/>
    <node id="intersection_3" x="-750" y="-250" type="traffic_light" tl="tls_3"/>
    <node id="intersection_4" x="-250" y="-250" type="traffic_light" tl="tls_4"/>

    <!-- Zone B (right 2x2 grid) -->
    <node id="intersection_5" x="250"  y="250"  type="traffic_light" tl="tls_5"/>
    <node id="intersection_6" x="750"  y="250"  type="traffic_light" tl="tls_6"/>
    <node id="intersection_7" x="250"  y="-250" type="traffic_light" tl="tls_7"/>
    <node id="intersection_8" x="750"  y="-250" type="traffic_light" tl="tls_8"/>

</nodes>
"""
    path = f'{OUTPUT_DIR}/federated.nod.xml'
    with open(path, 'w') as f:
        f.write(nodes_xml)
    print(f"  ✓ Nodes file: {path}")


def generate_edges():
    """Generate edge definitions for 4x2 grid with bidirectional roads"""
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">

    <!-- ==================== ZONE A (Intersections 1-4) ==================== -->

    <!-- Intersection 1 (Top-Left) external connections -->
    <edge id="north_to_i1" from="north_ext_1" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i1_to_north" from="intersection_1" to="north_ext_1" numLanes="2" speed="13.89"/>
    <edge id="west_to_i1"  from="west_ext_1"  to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i1_to_west"  from="intersection_1" to="west_ext_1"  numLanes="2" speed="13.89"/>

    <!-- Intersection 2 (Top-Middle-Left) external connections -->
    <edge id="north_to_i2" from="north_ext_2" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i2_to_north" from="intersection_2" to="north_ext_2" numLanes="2" speed="13.89"/>

    <!-- Intersection 3 (Bottom-Left) external connections -->
    <edge id="south_to_i3" from="south_ext_1" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="i3_to_south" from="intersection_3" to="south_ext_1" numLanes="2" speed="13.89"/>
    <edge id="west_to_i3"  from="west_ext_2"  to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="i3_to_west"  from="intersection_3" to="west_ext_2"  numLanes="2" speed="13.89"/>

    <!-- Intersection 4 (Bottom-Middle-Left) external connections -->
    <edge id="south_to_i4" from="south_ext_2" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="i4_to_south" from="intersection_4" to="south_ext_2" numLanes="2" speed="13.89"/>

    <!-- Zone A internal connections (bidirectional) -->
    <edge id="i1_to_i2" from="intersection_1" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i2_to_i1" from="intersection_2" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i1_to_i3" from="intersection_1" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="i3_to_i1" from="intersection_3" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i2_to_i4" from="intersection_2" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="i4_to_i2" from="intersection_4" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i3_to_i4" from="intersection_3" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="i4_to_i3" from="intersection_4" to="intersection_3" numLanes="2" speed="13.89"/>

    <!-- ==================== ZONE B (Intersections 5-8) ==================== -->

    <!-- Intersection 5 (Top-Middle-Right) external connections -->
    <edge id="north_to_i5" from="north_ext_3" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i5_to_north" from="intersection_5" to="north_ext_3" numLanes="2" speed="13.89"/>

    <!-- Intersection 6 (Top-Right) external connections -->
    <edge id="north_to_i6" from="north_ext_4" to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i6_to_north" from="intersection_6" to="north_ext_4" numLanes="2" speed="13.89"/>
    <edge id="east_to_i6"  from="east_ext_1"  to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i6_to_east"  from="intersection_6" to="east_ext_1"  numLanes="2" speed="13.89"/>

    <!-- Intersection 7 (Bottom-Middle-Right) external connections -->
    <edge id="south_to_i7" from="south_ext_3" to="intersection_7" numLanes="2" speed="13.89"/>
    <edge id="i7_to_south" from="intersection_7" to="south_ext_3" numLanes="2" speed="13.89"/>

    <!-- Intersection 8 (Bottom-Right) external connections -->
    <edge id="south_to_i8" from="south_ext_4" to="intersection_8" numLanes="2" speed="13.89"/>
    <edge id="i8_to_south" from="intersection_8" to="south_ext_4" numLanes="2" speed="13.89"/>
    <edge id="east_to_i8"  from="east_ext_2"  to="intersection_8" numLanes="2" speed="13.89"/>
    <edge id="i8_to_east"  from="intersection_8" to="east_ext_2"  numLanes="2" speed="13.89"/>

    <!-- Zone B internal connections (bidirectional) -->
    <edge id="i5_to_i6" from="intersection_5" to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i6_to_i5" from="intersection_6" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i5_to_i7" from="intersection_5" to="intersection_7" numLanes="2" speed="13.89"/>
    <edge id="i7_to_i5" from="intersection_7" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i6_to_i8" from="intersection_6" to="intersection_8" numLanes="2" speed="13.89"/>
    <edge id="i8_to_i6" from="intersection_8" to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i7_to_i8" from="intersection_7" to="intersection_8" numLanes="2" speed="13.89"/>
    <edge id="i8_to_i7" from="intersection_8" to="intersection_7" numLanes="2" speed="13.89"/>

    <!-- ==================== INTER-ZONE CONNECTIONS ==================== -->
    <!-- These are the critical links between Zone A and Zone B -->
    <!-- Top bridge: Intersection 2 <-> Intersection 5 -->
    <edge id="i2_to_i5" from="intersection_2" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i5_to_i2" from="intersection_5" to="intersection_2" numLanes="2" speed="13.89"/>

    <!-- Bottom bridge: Intersection 4 <-> Intersection 7 -->
    <edge id="i4_to_i7" from="intersection_4" to="intersection_7" numLanes="2" speed="13.89"/>
    <edge id="i7_to_i4" from="intersection_7" to="intersection_4" numLanes="2" speed="13.89"/>

</edges>
"""
    path = f'{OUTPUT_DIR}/federated.edg.xml'
    with open(path, 'w') as f:
        f.write(edges_xml)
    print(f"  ✓ Edges file: {path}")


def generate_traffic_lights():
    """Generate traffic light logic for all 8 intersections"""
    tls_entries = []
    for i in range(1, 9):
        tls_entries.append(f"""
    <tlLogic id="tls_{i}" type="static" programID="0" offset="0">
        <phase duration="60" state="GGrrGGrr"/>  <!-- NS green -->
        <phase duration="60" state="rrGGrrGG"/>  <!-- EW green -->
    </tlLogic>""")

    tls_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
{"".join(tls_entries)}

</additional>
"""
    path = f'{OUTPUT_DIR}/federated.tls.xml'
    with open(path, 'w') as f:
        f.write(tls_xml)
    print(f"  ✓ Traffic lights: {path}")


def generate_routes():
    """Generate traffic routes through the 4x2 grid network"""
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5"
           minGap="2.5" maxSpeed="50" guiShape="passenger"/>

    <!-- ======================== NORTH-SOUTH ROUTES ======================== -->
    <!-- Column 1: north_ext_1 -> i1 -> i3 -> south_ext_1 -->
    <route id="ns_col1" edges="north_to_i1 i1_to_i3 i3_to_south"/>
    <route id="sn_col1" edges="south_to_i3 i3_to_i1 i1_to_north"/>

    <!-- Column 2: north_ext_2 -> i2 -> i4 -> south_ext_2 -->
    <route id="ns_col2" edges="north_to_i2 i2_to_i4 i4_to_south"/>
    <route id="sn_col2" edges="south_to_i4 i4_to_i2 i2_to_north"/>

    <!-- Column 3: north_ext_3 -> i5 -> i7 -> south_ext_3 -->
    <route id="ns_col3" edges="north_to_i5 i5_to_i7 i7_to_south"/>
    <route id="sn_col3" edges="south_to_i7 i7_to_i5 i5_to_north"/>

    <!-- Column 4: north_ext_4 -> i6 -> i8 -> south_ext_4 -->
    <route id="ns_col4" edges="north_to_i6 i6_to_i8 i8_to_south"/>
    <route id="sn_col4" edges="south_to_i8 i8_to_i6 i6_to_north"/>

    <!-- ======================== EAST-WEST ROUTES ======================== -->
    <!-- Row 1 (top): west -> i1 -> i2 -> i5 -> i6 -> east  (full corridor) -->
    <route id="we_row1_full" edges="west_to_i1 i1_to_i2 i2_to_i5 i5_to_i6 i6_to_east"/>
    <route id="ew_row1_full" edges="east_to_i6 i6_to_i5 i5_to_i2 i2_to_i1 i1_to_west"/>

    <!-- Row 2 (bottom): west -> i3 -> i4 -> i7 -> i8 -> east -->
    <route id="we_row2_full" edges="west_to_i3 i3_to_i4 i4_to_i7 i7_to_i8 i8_to_east"/>
    <route id="ew_row2_full" edges="east_to_i8 i8_to_i7 i7_to_i4 i4_to_i3 i3_to_west"/>

    <!-- Row 1 Zone A only: west -> i1 -> i2 -> north/south exit -->
    <route id="we_row1_zoneA" edges="west_to_i1 i1_to_i2 i2_to_north"/>
    <!-- Row 2 Zone B only: south -> i7 -> i8 -> east -->
    <route id="sn_to_east" edges="south_to_i7 i7_to_i8 i8_to_east"/>

    <!-- ======================== CROSS-ZONE ROUTES ======================== -->
    <!-- Diagonal: northwest to southeast -->
    <route id="nw_to_se" edges="north_to_i1 i1_to_i2 i2_to_i5 i5_to_i7 i7_to_south"/>
    <!-- Diagonal: northeast to southwest -->
    <route id="ne_to_sw" edges="north_to_i6 i6_to_i5 i5_to_i2 i2_to_i4 i4_to_south"/>
    <!-- Zone A to Zone B via top bridge -->
    <route id="zA_to_zB_top" edges="west_to_i1 i1_to_i2 i2_to_i5 i5_to_i6 i6_to_east"/>
    <!-- Zone B to Zone A via bottom bridge -->
    <route id="zB_to_zA_bot" edges="east_to_i8 i8_to_i7 i7_to_i4 i4_to_i3 i3_to_west"/>

    <!-- ======================== LOCAL ROUTES (short trips) ======================== -->
    <route id="n1_to_e_via_zoneB" edges="north_to_i1 i1_to_i2 i2_to_i5 i5_to_i6 i6_to_east"/>
    <route id="s3_to_n3" edges="south_to_i7 i7_to_i5 i5_to_north"/>
    <route id="w_to_s_col1" edges="west_to_i1 i1_to_i3 i3_to_south"/>
    <route id="e_to_n_col4" edges="east_to_i6 i6_to_north"/>

    <!-- ======================== TRAFFIC FLOWS ======================== -->
    <!-- North-South flows (all 4 columns) -->
    <flow id="f_ns1" type="car" route="ns_col1" begin="0" end="3600" probability="0.05"/>
    <flow id="f_ns2" type="car" route="ns_col2" begin="0" end="3600" probability="0.05"/>
    <flow id="f_ns3" type="car" route="ns_col3" begin="0" end="3600" probability="0.05"/>
    <flow id="f_ns4" type="car" route="ns_col4" begin="0" end="3600" probability="0.05"/>
    <flow id="f_sn1" type="car" route="sn_col1" begin="0" end="3600" probability="0.05"/>
    <flow id="f_sn2" type="car" route="sn_col2" begin="0" end="3600" probability="0.05"/>
    <flow id="f_sn3" type="car" route="sn_col3" begin="0" end="3600" probability="0.05"/>
    <flow id="f_sn4" type="car" route="sn_col4" begin="0" end="3600" probability="0.05"/>

    <!-- East-West full corridor flows -->
    <flow id="f_we1" type="car" route="we_row1_full" begin="0" end="3600" probability="0.04"/>
    <flow id="f_we2" type="car" route="we_row2_full" begin="0" end="3600" probability="0.04"/>
    <flow id="f_ew1" type="car" route="ew_row1_full" begin="0" end="3600" probability="0.04"/>
    <flow id="f_ew2" type="car" route="ew_row2_full" begin="0" end="3600" probability="0.04"/>

    <!-- Cross-zone flows (moderate) -->
    <flow id="f_nw_se"  type="car" route="nw_to_se"  begin="0" end="3600" probability="0.03"/>
    <flow id="f_ne_sw"  type="car" route="ne_to_sw"  begin="0" end="3600" probability="0.03"/>
    <flow id="f_zA_zB"  type="car" route="zA_to_zB_top"  begin="0" end="3600" probability="0.03"/>
    <flow id="f_zB_zA"  type="car" route="zB_to_zA_bot"  begin="0" end="3600" probability="0.03"/>

    <!-- Local/short flows -->
    <flow id="f_local1" type="car" route="s3_to_n3"    begin="0" end="3600" probability="0.03"/>
    <flow id="f_local2" type="car" route="w_to_s_col1" begin="0" end="3600" probability="0.03"/>
    <flow id="f_local3" type="car" route="sn_to_east"  begin="0" end="3600" probability="0.03"/>
    <flow id="f_local4" type="car" route="e_to_n_col4" begin="0" end="3600" probability="0.03"/>

</routes>
"""
    path = f'{OUTPUT_DIR}/federated.rou.xml'
    with open(path, 'w') as f:
        f.write(routes_xml)
    print(f"  ✓ Routes file: {path}")


def generate_sumocfg():
    """Generate SUMO configuration file"""
    cfg_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="federated.net.xml"/>
        <route-files value="federated.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

</configuration>
"""
    path = f'{OUTPUT_DIR}/federated.sumocfg'
    with open(path, 'w') as f:
        f.write(cfg_xml)
    print(f"  ✓ SUMO config: {path}")


def build_network():
    """Build SUMO network using netconvert"""
    print("\nBuilding SUMO network with netconvert...")
    cmd = [
        'netconvert',
        f'--node-files={OUTPUT_DIR}/federated.nod.xml',
        f'--edge-files={OUTPUT_DIR}/federated.edg.xml',
        f'--output-file={OUTPUT_DIR}/federated.net.xml',
        '--no-turnarounds=true'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Network built successfully!")
        else:
            print(f"  ⚠ Warning: {result.stderr}")
    except Exception as e:
        print(f"  ⚠ Could not build network: {e}")
        print("  Run manually: " + " ".join(cmd))


def generate_all():
    """Generate all federated SUMO files"""
    print("=" * 70)
    print("Generating Federated SUMO Network (4x2 Grid - 8 Intersections)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_nodes()
    generate_edges()
    generate_traffic_lights()
    generate_routes()
    generate_sumocfg()
    build_network()

    print("\n" + "=" * 70)
    print("✅ Federated Network Generation Complete!")
    print("=" * 70)
    print()
    print("  Network Layout (4x2 Grid):")
    print()
    print("       Zone A (Supervisor 1)     Zone B (Supervisor 2)")
    print("      ┌───────────────────┐     ┌───────────────────┐")
    print("      │ [TLS1] ── [TLS2]──┼─────┼──[TLS5] ── [TLS6]│")
    print("      │   |         |     │     │    |         |    │")
    print("      │ [TLS3] ── [TLS4]──┼─────┼──[TLS7] ── [TLS8]│")
    print("      └───────────────────┘     └───────────────────┘")
    print()
    print(f"  Files created in: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    generate_all()
