from collections import deque
from typing import Deque, List

import numpy as np

from random import shuffle

import simpy

from sim_src.sim_env.env_object import *

class Packet():
    def __init__(self, length_bits, gen_time_us):
        self.hops_addr = []
        self.gen_time_us = gen_time_us
        self.end_time_us = -1
        self.hops_arrive_time_us = []
        self.hops_depart_time_us = []

        self.length_bits = length_bits
        
    def arrive_at(self, addr, cur_time_us):
        self.hops_addr.append(addr)
        self.hops_arrive_time_us.append(cur_time_us)
        self.hops_depart_time_us.append(-1)

    def depart_at(self, addr, cur_time_us):
        assert self.hops_addr[-1] == addr
        self.hops_depart_time_us[-1] = cur_time_us
        
    def end(self,cur_time_us):
        self.end_time_us = cur_time_us
        return self.get_tot_time_us()
        
    def get_last_hop_arr_time_us(self):
        return self.hops_arrive_time_us[-1]
    
    def get_gen_time_us(self):
        return self.gen_time_us
    
    def get_tot_time_us(self):
        assert self.end_time_us >= 0
        return self.end_time_us - self.gen_time_us
    
class P2PLinkInterfaceRx():
    def push(self, packets):
        pass

class Queue(P2PLinkInterfaceRx):
    def __init__(self, id=0, max_size=10000):
        self.id = id
        self.n_packet = 0
        self.queue:Deque[Packet] = deque(maxlen=max_size)

    def get_hol_vs_hop(self, cur_time_us):
        if self.queue:
            return self.queue[0].get_last_hop_arr_time_us(self)
        else:
            return 0

    def get_hol_vs_src(self, cur_time_us):
        if self.queue:
            return cur_time_us - self.queue[0].get_gen_time_us(self)
        else:
            return 0
    
    def pop(self, n=1):
        ret = []
        while self.queue and n > 0:
            ret.append(self.queue.popleft())
            n -= 1
        return ret
    
    def push(self, packets):
        n_discard = 0
        for p in packets:
            if len(self.queue) == self.queue.maxlen:
                n_discard += 1
            else:
                self.queue.append(p)
        return n_discard
    
    def get_bits_total(self):
        ret = 0
        for x in range(len(self.queue)):
            ret += self.queue[x].length_bits

        return ret 

    def get_pkts_total(self):
        return len(self.queue)
    
    def is_empty(self):
        return len(self.queue) == 0 
    
    
class PacketSrc(EnvObjectRunnable):
    def __init__(self, to_node=None, packet_length_bits=5e3, packet_rate_per_interval = 1000, packet_generation_interval_us = 1e3):
        super().__init__()
        self.packet_length_bits = packet_length_bits
        self.packet_rate_per_interval = packet_rate_per_interval
        self.packet_generation_interval_us = packet_generation_interval_us

        self.to_node = to_node

    def set_packet_rate_per_interval(self,packet_rate_per_interval):
        self.packet_rate_per_interval = packet_rate_per_interval

    def run(self):
        while True:
            packets = self._gen_packets()
            self.to_node.push(packets)
            yield self.env.timeout(self.packet_generation_interval_us)            

    def _gen_packets(self):
        num_packets = np.random.poisson(self.packet_rate_per_interval)
        packets = [Packet(self.packet_length_bits,self.env.now) for i in range(num_packets)]
        return packets

class PacketDst(EnvObject,P2PLinkInterfaceRx):
    def __init__(self):
        self.packet_counter = 0
        self.pdelay_counter = 0.
        self.pltbit_counter = 0
        
    def push(self, packets:List[Packet]):
        self.packet_counter += len(packets)
        for p in packets:
            self.pdelay_counter += p.end(self.env.now) 
            self.pltbit_counter += p.length_bits
            
            
class P2PLinkDirtTo(EnvObject,P2PLinkInterfaceRx):
    def __init__(self, to_node=None, bits_per_us=1e4, delay_us=100):
        self.to_node = to_node
        self.bits_per_us = bits_per_us
        self.delay_us = delay_us
        
        self.n_discard = 0
        self.current_load_bits = 0 
        
    def push(self, packets):
        self.current_load_bits
        
        for p in packets:
            tx_time_us = p.length_bits / self.bits_per_us
            if self.current_load_bits + p.length_bits > self.bits_per_us * tx_time_us:
                self.n_discard += 1
                continue
            else:
                self.env.process(self.push_a_packet(p,tx_time_us))

    def push_a_packet(self, p:Packet, tx_time_us):
        self.current_load_bits += p.length_bits
        yield self.env.timeout(self.delay_us + tx_time_us)
        self.to_node.push([p])
        self.current_load_bits -= p.length_bits

class NodeWithLocalPacketSrcDst_Base(EnvObjectRunnable,P2PLinkInterfaceRx):
    def __init__(self, id = 0, queue_size=10, packet_rate_per_ms = 5, packet_pop_interval_us = 1000):
        super().__init__()
        self.id = id
        self.packet_src = PacketSrc(to_node=self)
        self.packet_dst = PacketDst()
        self.packet_que = Queue(max_size=queue_size)
        
        self.packet_rate_per_ms = packet_rate_per_ms
        self.packet_pop_interval_us = packet_pop_interval_us
        self.p2plink_to_neighbor = []

    def run(self):
        self.env.process(self.queueing())
        yield self.env.event()


    def queueing(self):
        while True:
            # print(self.packet_dst.packet_counter,self.packet_dst.pdelay_counter,self.packet_dst.pdelay_counter/(self.packet_dst.packet_counter+1))
            yield self.env.timeout(self.packet_pop_interval_us)
            self.packet_dst.push(self.packet_que.pop(self.packet_rate_per_ms))

    def push(self, packets):
        pass


if __name__ == "__main__":
    class NodeWithLocalPacketSrcDst_BaseTest(NodeWithLocalPacketSrcDst_Base):
        def push(self, packets):
            n_discard = self.packet_que.push(packets)
            for i in self.p2plink_to_neighbor:
                i.push(packets)        
    
    # SimPy environment setup
    env = simpy.Environment()
    EnvObject.set_env(env=env)
    n_a = NodeWithLocalPacketSrcDst_BaseTest()
    n_b = NodeWithLocalPacketSrcDst_BaseTest()
    n_a.p2plink_to_neighbor.append(P2PLinkDirtTo(n_b))
    env.run(until=1000000)
    print(env.now,n_b.get_run_time_us())

    print(n_b.packet_dst.packet_counter)
    