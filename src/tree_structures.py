from typing import Dict, List, Set

class Node:
    def __init__(self,name:str,desc:str,index:int,children:Set[int],embeddings) -> None:
        self.name = name
        self.text = desc
        self.index = index
        self.children = children
        self.embeddings = embeddings

class Tree:
    def __init__(self,all_nodes,root_nodes,leaf_nodes,num_layers,lay_to_nodes) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = lay_to_nodes