"""
Generate SUMO Network for 8-Intersection Hierarchical Multi-Agent Traffic Control
2x4 Grid: Two cooperative groups of 4 intersections each

Layout (500m spacing):
  Group A (Left)           Group B (Right)
  [TLS_1]---[TLS_2]  <->  [TLS_5]---[TLS_6]
     |         |              |         |
  [TLS_3]---[TLS_4]  <->  [TLS_7]---[TLS_8]

Border links: TLS_2<->TLS_5 (top), TLS_4<->TLS_7 (bottom)
"""

import os
import subprocess


def generate_nodes():
    """Generate node definitions for 2x4 grid network"""
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">

    <!-- External boundary nodes -->
    <!-- North boundary (above row 1) -->
    <node id="north_ext_1" x="-750" y="750" type="priority"/>
    <node id="north_ext_2" x="-250" y="750" type="priority"/>
    <node id="north_ext_3" x="250"  y="750" type="priority"/>
    <node id="north_ext_4" x="750"  y="750" type="priority"/>

    <!-- South boundary (below row 2) -->
    <node id="south_ext_1" x="-750" y="-750" type="priority"/>
    <node id="south_ext_2" x="-250" y="-750" type="priority"/>
    <node id="south_ext_3" x="250"  y="-750" type="priority"/>
    <node id="south_ext_4" x="750"  y="-750" type="priority"/>

    <!-- West boundary (left of column 1) -->
    <node id="west_ext_1" x="-1250" y="250"  type="priority"/>
    <node id="west_ext_2" x="-1250" y="-250" type="priority"/>

    <!-- East boundary (right of column 4) -->
    <node id="east_ext_1" x="1250" y="250"  type="priority"/>
    <node id="east_ext_2" x="1250" y="-250" type="priority"/>

    <!-- Intersection nodes (traffic light controlled) -->
    <!-- Group A (Left 2x2) -->
    <node id="intersection_1" x="-750" y="250"  type="traffic_light" tl="tls_1"/>
    <node id="intersection_2" x="-250" y="250"  type="traffic_light" tl="tls_2"/>
    <node id="intersection_3" x="-750" y="-250" type="traffic_light" tl="tls_3"/>
    <node id="intersection_4" x="-250" y="-250" type="traffic_light" tl="tls_4"/>

    <!-- Group B (Right 2x2) -->
    <node id="intersection_5" x="250"  y="250"  type="traffic_light" tl="tls_5"/>
    <node id="intersection_6" x="750"  y="250"  type="traffic_light" tl="tls_6"/>
    <node id="intersection_7" x="250"  y="-250" type="traffic_light" tl="tls_7"/>
    <node id="intersection_8" x="750"  y="-250" type="traffic_light" tl="tls_8"/>

</nodes>
"""
    with open('sumo_files/grid8/network.nod.xml', 'w') as f:
        f.write(nodes_xml)
    print("✓ Nodes file generated")


def generate_edges():
    """Generate edge definitions for 2x4 grid"""
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">

    <!-- ========== INTERSECTION 1 (Top-Left, Group A) ========== -->
    <edge id="north_to_i1" from="north_ext_1" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i1_to_north" from="intersection_1" to="north_ext_1" numLanes="2" speed="13.89"/>
    <edge id="west_to_i1"  from="west_ext_1"  to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i1_to_west"  from="intersection_1" to="west_ext_1"  numLanes="2" speed="13.89"/>
    <edge id="i1_to_i2"    from="intersection_1" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i1_to_i3"    from="intersection_1" to="intersection_3" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 2 (Top-Center-Left, Group A) ========== -->
    <edge id="north_to_i2" from="north_ext_2" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i2_to_north" from="intersection_2" to="north_ext_2" numLanes="2" speed="13.89"/>
    <edge id="i2_to_i1"    from="intersection_2" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i2_to_i4"    from="intersection_2" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="i2_to_i5"    from="intersection_2" to="intersection_5" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 3 (Bottom-Left, Group A) ========== -->
    <edge id="south_to_i3" from="south_ext_1" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="i3_to_south" from="intersection_3" to="south_ext_1" numLanes="2" speed="13.89"/>
    <edge id="west_to_i3"  from="west_ext_2"  to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="i3_to_west"  from="intersection_3" to="west_ext_2"  numLanes="2" speed="13.89"/>
    <edge id="i3_to_i1"    from="intersection_3" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="i3_to_i4"    from="intersection_3" to="intersection_4" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 4 (Bottom-Center-Left, Group A) ========== -->
    <edge id="south_to_i4" from="south_ext_2" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="i4_to_south" from="intersection_4" to="south_ext_2" numLanes="2" speed="13.89"/>
    <edge id="i4_to_i2"    from="intersection_4" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i4_to_i3"    from="intersection_4" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="i4_to_i7"    from="intersection_4" to="intersection_7" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 5 (Top-Center-Right, Group B) ========== -->
    <edge id="north_to_i5" from="north_ext_3" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i5_to_north" from="intersection_5" to="north_ext_3" numLanes="2" speed="13.89"/>
    <edge id="i5_to_i2"    from="intersection_5" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="i5_to_i6"    from="intersection_5" to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i5_to_i7"    from="intersection_5" to="intersection_7" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 6 (Top-Right, Group B) ========== -->
    <edge id="north_to_i6" from="north_ext_4" to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i6_to_north" from="intersection_6" to="north_ext_4" numLanes="2" speed="13.89"/>
    <edge id="east_to_i6"  from="east_ext_1"  to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i6_to_east"  from="intersection_6" to="east_ext_1"  numLanes="2" speed="13.89"/>
    <edge id="i6_to_i5"    from="intersection_6" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i6_to_i8"    from="intersection_6" to="intersection_8" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 7 (Bottom-Center-Right, Group B) ========== -->
    <edge id="south_to_i7" from="south_ext_3" to="intersection_7" numLanes="2" speed="13.89"/>
    <edge id="i7_to_south" from="intersection_7" to="south_ext_3" numLanes="2" speed="13.89"/>
    <edge id="i7_to_i4"    from="intersection_7" to="intersection_4" numLanes="2" speed="13.89"/>
    <edge id="i7_to_i5"    from="intersection_7" to="intersection_5" numLanes="2" speed="13.89"/>
    <edge id="i7_to_i8"    from="intersection_7" to="intersection_8" numLanes="2" speed="13.89"/>

    <!-- ========== INTERSECTION 8 (Bottom-Right, Group B) ========== -->
    <edge id="south_to_i8" from="south_ext_4" to="intersection_8" numLanes="2" speed="13.89"/>
    <edge id="i8_to_south" from="intersection_8" to="south_ext_4" numLanes="2" speed="13.89"/>
    <edge id="east_to_i8"  from="east_ext_2"  to="intersection_8" numLanes="2" speed="13.89"/>
    <edge id="i8_to_east"  from="intersection_8" to="east_ext_2"  numLanes="2" speed="13.89"/>
    <edge id="i8_to_i6"    from="intersection_8" to="intersection_6" numLanes="2" speed="13.89"/>
    <edge id="i8_to_i7"    from="intersection_8" to="intersection_7" numLanes="2" speed="13.89"/>

</edges>
"""
    with open('sumo_files/grid8/network.edg.xml', 'w') as f:
        f.write(edges_xml)
    print("✓ Edges file generated")


def generate_routes():
    """Generate traffic routes for 2x4 grid network"""
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5"
           minGap="2.5" maxSpeed="50" guiShape="passenger"/>

    <!-- ====== NORTH-SOUTH through routes (vertical) ====== -->
    <route id="n1_s1" edges="north_to_i1 i1_to_i3 i3_to_south"/>
    <route id="n2_s2" edges="north_to_i2 i2_to_i4 i4_to_south"/>
    <route id="n3_s3" edges="north_to_i5 i5_to_i7 i7_to_south"/>
    <route id="n4_s4" edges="north_to_i6 i6_to_i8 i8_to_south"/>

    <route id="s1_n1" edges="south_to_i3 i3_to_i1 i1_to_north"/>
    <route id="s2_n2" edges="south_to_i4 i4_to_i2 i2_to_north"/>
    <route id="s3_n3" edges="south_to_i7 i7_to_i5 i5_to_north"/>
    <route id="s4_n4" edges="south_to_i8 i8_to_i6 i6_to_north"/>

    <!-- ====== WEST-EAST through routes (full horizontal) ====== -->
    <route id="w1_e1" edges="west_to_i1 i1_to_i2 i2_to_i5 i5_to_i6 i6_to_east"/>
    <route id="w2_e2" edges="west_to_i3 i3_to_i4 i4_to_i7 i7_to_i8 i8_to_east"/>

    <route id="e1_w1" edges="east_to_i6 i6_to_i5 i5_to_i2 i2_to_i1 i1_to_west"/>
    <route id="e2_w2" edges="east_to_i8 i8_to_i7 i7_to_i4 i4_to_i3 i3_to_west"/>

    <!-- ====== CROSS-GROUP routes (diagonal / turning) ====== -->
    <route id="n1_e1" edges="north_to_i1 i1_to_i2 i2_to_i5 i5_to_i6 i6_to_east"/>
    <route id="n2_s3" edges="north_to_i2 i2_to_i5 i5_to_i7 i7_to_south"/>
    <route id="w1_s2" edges="west_to_i1 i1_to_i3 i3_to_i4 i4_to_south"/>
    <route id="e1_n3" edges="east_to_i6 i6_to_i5 i5_to_north"/>
    <route id="s4_w2" edges="south_to_i8 i8_to_i7 i7_to_i4 i4_to_i3 i3_to_west"/>
    <route id="s1_e2" edges="south_to_i3 i3_to_i4 i4_to_i7 i7_to_i8 i8_to_east"/>

    <!-- ====== LOCAL routes (within groups) ====== -->
    <!-- Group A local -->
    <route id="n1_w1" edges="north_to_i1 i1_to_west"/>
    <route id="w1_s1" edges="west_to_i1 i1_to_i3 i3_to_south"/>
    <route id="n2_s1_cross" edges="north_to_i2 i2_to_i4 i4_to_i3 i3_to_south"/>

    <!-- Group B local -->
    <route id="n4_e1" edges="north_to_i6 i6_to_east"/>
    <route id="e2_s4" edges="east_to_i8 i8_to_south"/>
    <route id="n3_e1_cross" edges="north_to_i5 i5_to_i6 i6_to_east"/>

    <!-- ====== Traffic flows ====== -->
    <!-- North-South flows (balanced per column) -->
    <flow id="f_n1_s1" type="car" route="n1_s1" begin="0" end="3600" probability="0.04"/>
    <flow id="f_n2_s2" type="car" route="n2_s2" begin="0" end="3600" probability="0.04"/>
    <flow id="f_n3_s3" type="car" route="n3_s3" begin="0" end="3600" probability="0.04"/>
    <flow id="f_n4_s4" type="car" route="n4_s4" begin="0" end="3600" probability="0.04"/>

    <flow id="f_s1_n1" type="car" route="s1_n1" begin="0" end="3600" probability="0.04"/>
    <flow id="f_s2_n2" type="car" route="s2_n2" begin="0" end="3600" probability="0.04"/>
    <flow id="f_s3_n3" type="car" route="s3_n3" begin="0" end="3600" probability="0.04"/>
    <flow id="f_s4_n4" type="car" route="s4_n4" begin="0" end="3600" probability="0.04"/>

    <!-- West-East flows (full through-traffic) -->
    <flow id="f_w1_e1" type="car" route="w1_e1" begin="0" end="3600" probability="0.04"/>
    <flow id="f_w2_e2" type="car" route="w2_e2" begin="0" end="3600" probability="0.04"/>
    <flow id="f_e1_w1" type="car" route="e1_w1" begin="0" end="3600" probability="0.04"/>
    <flow id="f_e2_w2" type="car" route="e2_w2" begin="0" end="3600" probability="0.04"/>

    <!-- Cross-group flows (moderate) -->
    <flow id="f_n1_e1" type="car" route="n1_e1" begin="0" end="3600" probability="0.02"/>
    <flow id="f_n2_s3" type="car" route="n2_s3" begin="0" end="3600" probability="0.02"/>
    <flow id="f_w1_s2" type="car" route="w1_s2" begin="0" end="3600" probability="0.02"/>
    <flow id="f_e1_n3" type="car" route="e1_n3" begin="0" end="3600" probability="0.02"/>
    <flow id="f_s4_w2" type="car" route="s4_w2" begin="0" end="3600" probability="0.02"/>
    <flow id="f_s1_e2" type="car" route="s1_e2" begin="0" end="3600" probability="0.02"/>

    <!-- Local flows -->
    <flow id="f_n1_w1" type="car" route="n1_w1" begin="0" end="3600" probability="0.02"/>
    <flow id="f_w1_s1" type="car" route="w1_s1" begin="0" end="3600" probability="0.02"/>
    <flow id="f_n4_e1" type="car" route="n4_e1" begin="0" end="3600" probability="0.02"/>
    <flow id="f_e2_s4" type="car" route="e2_s4" begin="0" end="3600" probability="0.02"/>
    <flow id="f_n2_s1_cross" type="car" route="n2_s1_cross" begin="0" end="3600" probability="0.02"/>
    <flow id="f_n3_e1_cross" type="car" route="n3_e1_cross" begin="0" end="3600" probability="0.02"/>

</routes>
"""
    with open('sumo_files/grid8/network.rou.xml', 'w') as f:
        f.write(routes_xml)
    print("✓ Routes file generated")


def generate_sumocfg():
    """Generate SUMO configuration file"""
    cfg_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="network.rou.xml"/>
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
    with open('sumo_files/grid8/network.sumocfg', 'w') as f:
        f.write(cfg_xml)
    print("✓ SUMO configuration generated")


def build_network():
    """Build SUMO network using netconvert"""
    print("\nBuilding SUMO network with netconvert...")
    cmd = [
        'netconvert',
        '--node-files=sumo_files/grid8/network.nod.xml',
        '--edge-files=sumo_files/grid8/network.edg.xml',
        '--output-file=sumo_files/grid8/network.net.xml',
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
    """Generate all 8-intersection SUMO files"""
    print("=" * 70)
    print("Generating 8-Intersection SUMO Network (2x4 Grid)")
    print("  Group A: TLS_1, TLS_2, TLS_3, TLS_4 (left)")
    print("  Group B: TLS_5, TLS_6, TLS_7, TLS_8 (right)")
    print("  Border:  TLS_2<->TLS_5 (top), TLS_4<->TLS_7 (bottom)")
    print("=" * 70)

    os.makedirs('sumo_files/grid8', exist_ok=True)

    generate_nodes()
    generate_edges()
    generate_routes()
    generate_sumocfg()
    build_network()

    print("\n" + "=" * 70)
    print("✅ 8-Intersection Network Generation Complete!")
    print("=" * 70)
    print("\nLayout:")
    print("  [TLS_1]---[TLS_2] <-> [TLS_5]---[TLS_6]")
    print("     |         |           |         |")
    print("  [TLS_3]---[TLS_4] <-> [TLS_7]---[TLS_8]")
    print("   Group A (left)       Group B (right)")
    print("\nFiles: sumo_files/grid8/")
    print("=" * 70)


if __name__ == '__main__':
    generate_all()
