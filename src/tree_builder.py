import copy
import tiktoken
from abc import abstractclassmethod
from typing import Dict,List,Optional,Set,Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding_model import OpenAIEmbeddingModel
from embedding_model import BaseEmbeddingModel
from summarization_model import GPT3TurboSummarizationModel
from summarization_model import BaseSummarizationModel
from logger import get_logger
from tree_structures import Node,Tree
from tqdm import tqdm
from graphviz import Digraph

logger = get_logger(__name__)

class TreeBuilderConfig:
    def __init__(self,tokenizer=None,max_tokens=None,num_layers=None,threshold=None,top_k=None,selection_mode=None,summarization_length=None,summarization_model=None,embedding_models = None,cluster_embedding_model=None):
        """
        tokenizer: 文本分词的编码器
        max_tokens: 最大令牌数
        num_layers: 树结构的层数
        threshold: 决策的阈值
        top_k: 
        selection_mode: 
        summarization_length: 摘要的长度
        summarization_model: 用于文本摘要的模型
        embedding_models: 用于嵌入的模型名称
        cluster_embedding_model: 用于聚类的嵌入模型名称
        """
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens,int) or max_tokens <1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens
        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers,int) or num_layers <1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers =  num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold,(int,float)) or not (0<= threshold <=1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold
        if top_k is None:
            top_k = 5
        if not isinstance(top_k,int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k =  top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model is None:
            summarization_model = GPT3TurboSummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        self.summarization_model = summarization_model

        if embedding_models is None:
            embedding_models = {"OpenAI": OpenAIEmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError(
                "embedding_models must be a dictionary of model_name: instance pairs"
            )
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError(
                    "All embedding models must be an instance of BaseEmbeddingModel"
                )
        self.embedding_models = embedding_models

        if cluster_embedding_model is None:
            cluster_embedding_model = "OpenAI"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )
        self.cluster_embedding_model = cluster_embedding_model
    
    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log

class TreeBuilder:
    def __init__(self, config) -> None:
        """Initialization"""
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        logger.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )

    def create_node(self,index:int,group:Dict,children_indices:Optional[Set[int]] = None) -> Tuple[int,Node]:
        "creates a new node with the given index, text, and children indices"
        if children_indices is None:
            children_indices = set()
        group_name, group_desc = group['name'], group["description"]
        embeddings = {
            model_name: model.create_embedding(group_desc)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(group_name,group_desc, index, children_indices, embeddings))
    
    def create_embedding(self,text) -> List[float]:
        "Generates embeddings for the given text using the specified embedding model"
        return self.embedding_models[self.cluster_embedding_model].create_embedding(text)
    
    def summarzie(self,context,max_tokens=150) -> str:
        "Generates a summary of the input context using the specified summarization model"
        return self.summarization_model.summarize(context,max_tokens)
    
    def multithreaded_create_leaf_nodes(self,groups:List[str]) -> Dict[int,Node]:
        "Creates leaf nodes using multithreading from the given list of next group"
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(groups)
            }
            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node
        return leaf_nodes
    
    def build_from_group(self,groups:str,use_multithreading:bool=False) -> Tree:
        "Builds a golen tree from the input text, optionally using multithreading"
        logger.info("Creating Leaf Nodes")
        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(groups)
        else:
            leaf_nodes = {}
            for index, group in tqdm(enumerate(groups), total=len(groups)):
                _,node = self.create_node(index,group)
                leaf_nodes[index] = node
        layer_to_nodes = {0:list(leaf_nodes.values())}
        logger.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logger.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(all_nodes,all_nodes,layer_to_nodes)
        tree = Tree(all_nodes,root_nodes,leaf_nodes,self.num_layers,layer_to_nodes)
        return tree
    
    @abstractclassmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
        of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.

        Args:
            current_level_nodes (Dict[int, Node]): The current set of nodes.
            all_tree_nodes (Dict[int, Node]): The dictionary of all nodes.
            use_multithreading (bool): Whether to use multithreading to speed up the process.

        Returns:
            Dict[int, Node]: The final set of root nodes.
        """
        pass


    def show_tree(tree:Tree):
        pass


    
