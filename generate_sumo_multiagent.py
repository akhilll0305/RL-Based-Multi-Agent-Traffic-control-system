"""
Generate SUMO Network for Multi-Agent Traffic Control (2x2 Grid)
4 Intersections with connecting roads
"""

import os


def generate_nodes():
    """Generate node definitions for 2x2 grid network"""
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    
    <!-- External boundary nodes (traffic sources/sinks) -->
    <!-- North boundary -->
    <node id="north_ext_1" x="-250" y="750" type="priority"/>
    <node id="north_ext_2" x="250" y="750" type="priority"/>
    
    <!-- South boundary -->
    <node id="south_ext_1" x="-250" y="-750" type="priority"/>
    <node id="south_ext_2" x="250" y="-750" type="priority"/>
    
    <!-- East boundary -->
    <node id="east_ext_1" x="750" y="250" type="priority"/>
    <node id="east_ext_2" x="750" y="-250" type="priority"/>
    
    <!-- West boundary -->
    <node id="west_ext_1" x="-750" y="250" type="priority"/>
    <node id="west_ext_2" x="-750" y="-250" type="priority"/>
    
    <!-- Intersection nodes (traffic light controlled) -->
    <!-- Top row intersections -->
    <node id="intersection_1" x="-250" y="250" type="traffic_light" tl="tls_1"/>
    <node id="intersection_2" x="250" y="250" type="traffic_light" tl="tls_2"/>
    
    <!-- Bottom row intersections -->
    <node id="intersection_3" x="-250" y="-250" type="traffic_light" tl="tls_3"/>
    <node id="intersection_4" x="250" y="-250" type="traffic_light" tl="tls_4"/>
    
</nodes>
"""
    
    with open('sumo_config/multi_intersection/multiagent.nod.xml', 'w') as f:
        f.write(nodes_xml)
    print("✓ Nodes file generated")


def generate_edges():
    """Generate edge definitions connecting nodes"""
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    
    <!-- INTERSECTION 1 (Top-Left) connections -->
    <!-- From external to intersection 1 -->
    <edge id="north_to_i1" from="north_ext_1" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="west_to_i1" from="west_ext_1" to="intersection_1" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 1 to external -->
    <edge id="i1_to_north" from="intersection_1" to="north_ext_1" numLanes="2" speed="13.89"/>
    <edge id="i1_to_west" from="intersection_1" to="west_ext_1" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 1 to intersection 2 (east) -->
    <edge id="i1_to_i2" from="intersection_1" to="intersection_2" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 1 to intersection 3 (south) -->
    <edge id="i1_to_i3" from="intersection_1" to="intersection_3" numLanes="2" speed="13.89"/>
    
    
    <!-- INTERSECTION 2 (Top-Right) connections -->
    <!-- From external to intersection 2 -->
    <edge id="north_to_i2" from="north_ext_2" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="east_to_i2" from="east_ext_1" to="intersection_2" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 2 to external -->
    <edge id="i2_to_north" from="intersection_2" to="north_ext_2" numLanes="2" speed="13.89"/>
    <edge id="i2_to_east" from="intersection_2" to="east_ext_1" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 2 to intersection 1 (west) -->
    <edge id="i2_to_i1" from="intersection_2" to="intersection_1" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 2 to intersection 4 (south) -->
    <edge id="i2_to_i4" from="intersection_2" to="intersection_4" numLanes="2" speed="13.89"/>
    
    
    <!-- INTERSECTION 3 (Bottom-Left) connections -->
    <!-- From external to intersection 3 -->
    <edge id="south_to_i3" from="south_ext_1" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="west_to_i3" from="west_ext_2" to="intersection_3" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 3 to external -->
    <edge id="i3_to_south" from="intersection_3" to="south_ext_1" numLanes="2" speed="13.89"/>
    <edge id="i3_to_west" from="intersection_3" to="west_ext_2" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 3 to intersection 1 (north) -->
    <edge id="i3_to_i1" from="intersection_3" to="intersection_1" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 3 to intersection 4 (east) -->
    <edge id="i3_to_i4" from="intersection_3" to="intersection_4" numLanes="2" speed="13.89"/>
    
    
    <!-- INTERSECTION 4 (Bottom-Right) connections -->
    <!-- From external to intersection 4 -->
    <edge id="south_to_i4" from="south_ext_2" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="east_to_i4" from="east_ext_2" to="intersection_4" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 4 to external -->
    <edge id="i4_to_south" from="intersection_4" to="south_ext_2" numLanes="2" speed="13.89"/>
    <edge id="i4_to_east" from="intersection_4" to="east_ext_2" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 4 to intersection 3 (west) -->
    <edge id="i4_to_i3" from="intersection_4" to="intersection_3" numLanes="2" speed="13.89"/>
    
    <!-- From intersection 4 to intersection 2 (north) -->
    <edge id="i4_to_i2" from="intersection_4" to="intersection_2" numLanes="2" speed="13.89"/>
    
</edges>
"""
    
    with open('sumo_config/multi_intersection/multiagent.edg.xml', 'w') as f:
        f.write(edges_xml)
    print("✓ Edges file generated")


def generate_traffic_lights():
    """Generate traffic light logic for all 4 intersections"""
    tls_xml = """<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    
    <!-- Traffic Light Logic - Simple 2-phase for all intersections -->
    <!-- Phase 0: North-South Green, East-West Red -->
    <!-- Phase 1: East-West Green, North-South Red -->
    
    <!-- TLS 1 (Top-Left Intersection) -->
    <tlLogic id="tls_1" type="static" programID="0" offset="0">
        <phase duration="60" state="GGrrGGrr"/>  <!-- NS green -->
        <phase duration="60" state="rrGGrrGG"/>  <!-- EW green -->
    </tlLogic>
    
    <!-- TLS 2 (Top-Right Intersection) -->
    <tlLogic id="tls_2" type="static" programID="0" offset="0">
        <phase duration="60" state="GGrrGGrr"/>  <!-- NS green -->
        <phase duration="60" state="rrGGrrGG"/>  <!-- EW green -->
    </tlLogic>
    
    <!-- TLS 3 (Bottom-Left Intersection) -->
    <tlLogic id="tls_3" type="static" programID="0" offset="0">
        <phase duration="60" state="GGrrGGrr"/>  <!-- NS green -->
        <phase duration="60" state="rrGGrrGG"/>  <!-- EW green -->
    </tlLogic>
    
    <!-- TLS 4 (Bottom-Right Intersection) -->
    <tlLogic id="tls_4" type="static" programID="0" offset="0">
        <phase duration="60" state="GGrrGGrr"/>  <!-- NS green -->
        <phase duration="60" state="rrGGrrGG"/>  <!-- EW green -->
    </tlLogic>
    
</additional>
"""
    
    with open('sumo_config/multi_intersection/multiagent.tls.xml', 'w') as f:
        f.write(tls_xml)
    print("✓ Traffic light logic generated")


def generate_routes():
    """Generate traffic routes through the network"""
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Vehicle type definition -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50" guiShape="passenger"/>
    
    <!-- Routes through the network -->
    <!-- North to South routes -->
    <route id="n1_to_s1" edges="north_to_i1 i1_to_i3 i3_to_south"/>
    <route id="n2_to_s2" edges="north_to_i2 i2_to_i4 i4_to_south"/>
    
    <!-- South to North routes -->
    <route id="s1_to_n1" edges="south_to_i3 i3_to_i1 i1_to_north"/>
    <route id="s2_to_n2" edges="south_to_i4 i4_to_i2 i2_to_north"/>
    
    <!-- East to West routes -->
    <route id="e1_to_w1" edges="east_to_i2 i2_to_i1 i1_to_west"/>
    <route id="e2_to_w2" edges="east_to_i4 i4_to_i3 i3_to_west"/>
    
    <!-- West to East routes -->
    <route id="w1_to_e1" edges="west_to_i1 i1_to_i2 i2_to_east"/>
    <route id="w2_to_e2" edges="west_to_i3 i3_to_i4 i4_to_east"/>
    
    <!-- Diagonal routes (through all 4 intersections) -->
    <route id="nw_to_se" edges="north_to_i1 i1_to_i2 i2_to_i4 i4_to_south"/>
    <route id="ne_to_sw" edges="north_to_i2 i2_to_i1 i1_to_i3 i3_to_south"/>
    
    <!-- Local routes (exit at first or second intersection) -->
    <route id="n1_to_e" edges="north_to_i1 i1_to_i2 i2_to_east"/>
    <route id="w1_to_s" edges="west_to_i1 i1_to_i3 i3_to_south"/>
    <route id="s2_to_w" edges="south_to_i4 i4_to_i3 i3_to_west"/>
    <route id="e1_to_n" edges="east_to_i2 i2_to_i1 i1_to_north"/>
    
    <!-- Traffic flows - balanced distribution -->
    <flow id="flow_n1_s1" type="car" route="n1_to_s1" begin="0" end="3600" probability="0.05"/>
    <flow id="flow_n2_s2" type="car" route="n2_to_s2" begin="0" end="3600" probability="0.05"/>
    <flow id="flow_s1_n1" type="car" route="s1_to_n1" begin="0" end="3600" probability="0.05"/>
    <flow id="flow_s2_n2" type="car" route="s2_to_n2" begin="0" end="3600" probability="0.05"/>
    
    <flow id="flow_e1_w1" type="car" route="e1_to_w1" begin="0" end="3600" probability="0.05"/>
    <flow id="flow_e2_w2" type="car" route="e2_to_w2" begin="0" end="3600" probability="0.05"/>
    <flow id="flow_w1_e1" type="car" route="w1_to_e1" begin="0" end="3600" probability="0.05"/>
    <flow id="flow_w2_e2" type="car" route="w2_to_e2" begin="0" end="3600" probability="0.05"/>
    
    <flow id="flow_nw_se" type="car" route="nw_to_se" begin="0" end="3600" probability="0.03"/>
    <flow id="flow_ne_sw" type="car" route="ne_to_sw" begin="0" end="3600" probability="0.03"/>
    
    <flow id="flow_n1_e" type="car" route="n1_to_e" begin="0" end="3600" probability="0.03"/>
    <flow id="flow_w1_s" type="car" route="w1_to_s" begin="0" end="3600" probability="0.03"/>
    <flow id="flow_s2_w" type="car" route="s2_to_w" begin="0" end="3600" probability="0.03"/>
    <flow id="flow_e1_n" type="car" route="e1_to_n" begin="0" end="3600" probability="0.03"/>
    
</routes>
"""
    
    with open('sumo_config/multi_intersection/multiagent.rou.xml', 'w') as f:
        f.write(routes_xml)
    print("✓ Routes file generated")


def generate_sumocfg():
    """Generate SUMO configuration file"""
    cfg_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <input>
        <net-file value="multiagent.net.xml"/>
        <route-files value="multiagent.rou.xml"/>
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
    
    with open('sumo_config/multi_intersection/multiagent.sumocfg', 'w') as f:
        f.write(cfg_xml)
    print("✓ SUMO configuration generated")


def build_network():
    """Build SUMO network using netconvert"""
    import subprocess
    
    print("\nBuilding SUMO network...")
    cmd = [
        'netconvert',
        '--node-files=sumo_config/multi_intersection/multiagent.nod.xml',
        '--edge-files=sumo_config/multi_intersection/multiagent.edg.xml',
        '--output-file=sumo_config/multi_intersection/multiagent.net.xml',
        '--no-turnarounds=true'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Network built successfully!")
        else:
            print(f"⚠ Warning: {result.stderr}")
    except Exception as e:
        print(f"⚠ Could not build network automatically: {e}")
        print("Please run manually:")
        print(" ".join(cmd))


def generate_all():
    """Generate all multi-agent SUMO files"""
    print("="*70)
    print("Generating Multi-Agent SUMO Network (2x2 Grid - 4 Intersections)")
    print("="*70)
    
    # Create directory
    os.makedirs('sumo_config/multi_intersection', exist_ok=True)
    
    # Generate files
    generate_nodes()
    generate_edges()
    generate_traffic_lights()
    generate_routes()
    generate_sumocfg()
    
    # Build network
    build_network()
    
    print("\n" + "="*70)
    print("✅ Multi-Agent Network Generation Complete!")
    print("="*70)
    print("\nNetwork Layout:")
    print("  [Int 1] ←→ [Int 2]")
    print("     ↕         ↕")
    print("  [Int 3] ←→ [Int 4]")
    print("\nFiles created in: sumo_config/multi_intersection/")
    print("  - multiagent.net.xml (network)")
    print("  - multiagent.rou.xml (routes)")
    print("  - multiagent.tls.xml (traffic lights)")
    print("  - multiagent.sumocfg (configuration)")
    print("="*70)


if __name__ == '__main__':
    generate_all()
