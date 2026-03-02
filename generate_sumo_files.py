"""
SUMO Configuration Files Generator
"""

import os
import subprocess


def generate_nodes():
    """Generate SUMO nodes file"""
    
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="n_north" x="0.00" y="200.00" type="priority"/>
    <node id="n_south" x="0.00" y="-200.00" type="priority"/>
    <node id="n_east" x="200.00" y="0.00" type="priority"/>
    <node id="n_west" x="-200.00" y="0.00" type="priority"/>
    <node id="center" x="0.00" y="0.00" type="traffic_light"/>
</nodes>
"""
    
    with open('sumo_config/single_intersection/intersection.nod.xml', 'w') as f:
        f.write(nodes_xml)
    
    print("Nodes file created: sumo_config/single_intersection/intersection.nod.xml")


def generate_edges():
    """Generate SUMO edges file"""
    
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <edge id="north_in" from="n_north" to="center" numLanes="2" speed="13.89"/>
    <edge id="south_in" from="n_south" to="center" numLanes="2" speed="13.89"/>
    <edge id="east_in" from="n_east" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="n_west" to="center" numLanes="2" speed="13.89"/>
    
    <edge id="north_out" from="center" to="n_north" numLanes="2" speed="13.89"/>
    <edge id="south_out" from="center" to="n_south" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center" to="n_east" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="n_west" numLanes="2" speed="13.89"/>
</edges>
"""
    
    with open('sumo_config/single_intersection/intersection.edg.xml', 'w') as f:
        f.write(edges_xml)
    
    print("Edges file created: sumo_config/single_intersection/intersection.edg.xml")


def generate_intersection_network():
    """Generate SUMO network file using netconvert"""
    
    # First create node and edge files
    generate_nodes()
    generate_edges()
    
    # Use netconvert to create the network
    print("Running netconvert to generate network...")
    try:
        subprocess.run([
            'netconvert',
            '--node-files=sumo_config/single_intersection/intersection.nod.xml',
            '--edge-files=sumo_config/single_intersection/intersection.edg.xml',
            '--output-file=sumo_config/single_intersection/intersection.net.xml',
            '--tls.guess-signals', 'true'
        ], check=True, capture_output=True, text=True)
        print("Network file created: sumo_config/single_intersection/intersection.net.xml")
    except subprocess.CalledProcessError as e:
        print(f"Error running netconvert: {e}")
        print(f"stderr: {e.stderr}")
        raise


def generate_routes():
    """Generate SUMO route file with traffic flows"""
    
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <!-- Vehicle Type -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="50.0" guiShape="passenger"/>

    <!-- North-South Traffic -->
    <flow id="north_south" type="car" begin="0" end="3600" probability="0.25" from="north_in" to="south_out"/>
    <flow id="south_north" type="car" begin="0" end="3600" probability="0.25" from="south_in" to="north_out"/>

    <!-- East-West Traffic -->
    <flow id="east_west" type="car" begin="0" end="3600" probability="0.20" from="east_in" to="west_out"/>
    <flow id="west_east" type="car" begin="0" end="3600" probability="0.20" from="west_in" to="east_out"/>

</routes>
"""
    
    with open('sumo_config/single_intersection/routes.rou.xml', 'w') as f:
        f.write(routes_xml)
    
    print("Route file created: sumo_config/single_intersection/routes.rou.xml")


def generate_sumocfg():
    """Generate SUMO configuration file"""
    
    sumocfg = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>

</configuration>
"""
    
    with open('sumo_config/single_intersection/simulation.sumocfg', 'w') as f:
        f.write(sumocfg)
    
    print("Configuration file created: sumo_config/single_intersection/simulation.sumocfg")


def generate_all_sumo_files():
    """Generate all required SUMO files"""
    os.makedirs('sumo_config/single_intersection', exist_ok=True)
    
    print("Generating SUMO configuration files...")
    generate_intersection_network()
    generate_routes()
    generate_sumocfg()
    print("All SUMO files generated successfully!")


if __name__ == '__main__':
    generate_all_sumo_files()
