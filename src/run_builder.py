import json
from cluster_tree_builder import ClusterTreeBuilder
from cluster_tree_builder import ClusterTreeConfig

def load_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_group():
    file_path = "/home/jindm/project/HieGroupArch/cluster/0_layer.json"
    group_info = load_file(file_path)
    return group_info

def main():
    group_info = read_group()
    tree_builder_config = ClusterTreeConfig()
    ClusterTree = ClusterTreeBuilder(tree_builder_config)
    ClusterTree.build_from_group(group_info)
    
if __name__=="__main__":
    main()