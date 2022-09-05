# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:37:51 2022

@author: ASUS
"""

from .Scheduler import*
import numpy as np
class SimpleScheduler(Scheduler ):
    def _init_(self):
        super( )._init_()
        
    def selection(self):
        selectedContainerIDs  = []
        for hostID, host in enumerate(self.env.hostlist):
            if host.getCPU() > 70:
                containerIDs=self.env.getContainersOfHost(hostID)
                if containerIDs:
                    containerIPS = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
                    selectedContainerIDs.append(containerIDs[np.argmax(containerIPS)])
        
        return selectedContainerIDs

    def placement(self, containerIDs):
        decision= []
        for cid in containerIDs:
            scores = [self.env.stats.runSimpleSimulation([(cid,hostID)]) for hostID, _ in enumerate(self.env.hostlist)]
            decision.append((cid, np.argmin(scores)))
        return decision