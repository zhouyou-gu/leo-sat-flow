from sim_src.core.queue_system import NodeWithLocalPacketSrcDst_Base




class SatNodeTwoISL():
    def __init__(self) -> None:
        self.que_system = NodeWithLocalPacketSrcDst_Base()
        self.orb_system = None
        self.qos_metric = None
        self.agent = None    
        
    
        