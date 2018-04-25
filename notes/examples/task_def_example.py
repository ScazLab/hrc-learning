    def _task_def(self):
        gp_l1 = LeafCombination(PredAction
                                ('gatherparts_leg_1', self.NUM_FEATS, self.OBS_PROBS['gp_l1_probs']))
        self.leaves['gatherparts_leg_1'] = gp_l1
        gp_l2 = LeafCombination(PredAction
                                ('gatherparts_leg_2', self.NUM_FEATS, self.OBS_PROBS['gp_l2_probs']))
        self.leaves['gatherparts_leg_2'] = gp_l2
        gp_l3 = LeafCombination(PredAction
                                ('gatherparts_leg_3', self.NUM_FEATS, self.OBS_PROBS['gp_l3_probs']))
        self.leaves['gatherparts_leg_3'] = gp_l3
        gp_l4 = LeafCombination(PredAction
                                ('gatherparts_leg_4', self.NUM_FEATS, self.OBS_PROBS['gp_l4_probs']))
        self.leaves['gatherparts_leg_4'] = gp_l4
        gp_s = LeafCombination(PredAction
                               ('gatherparts_seat', self.NUM_FEATS, self.OBS_PROBS['gp_s_probs']))
        self.leaves['gatherparts_seat'] = gp_s
        ass_s = LeafCombination(PredAction
                                ('assemble_seat', self.NUM_FEATS, self.OBS_PROBS['ass_s_probs']))
        self.leaves['assemble_seat'] = ass_s
        gp_bl = LeafCombination(PredAction
                                ('gatherparts_back_left', self.NUM_FEATS, self.OBS_PROBS['gp_b_probs']))
        self.leaves['gatherparts_back_left'] = gp_bl
        gp_br = LeafCombination(PredAction
                                ('gatherparts_back_right', self.NUM_FEATS, self.OBS_PROBS['gp_b_probs']))
        self.leaves['gatherparts_back_right'] = gp_br
        gp_bt = LeafCombination(PredAction
                                ('gatherparts_back_top', self.NUM_FEATS, self.OBS_PROBS['gp_b_probs']))
        self.leaves['gatherparts_back_top'] = gp_bt
        ass_b = LeafCombination(PredAction
                                ('assemble_back', self.NUM_FEATS, self.OBS_PROBS['ass_b_probs']))
        self.leaves['assemble_back'] = ass_b
        f_legs = ParallelCombination([gp_l1, gp_l2, gp_l3, gp_l4], name='finish_legs')
        f_base = SequentialCombination([f_legs, gp_s, ass_s], name='finish_base')
        gp_back_low = ParallelCombination([gp_bl, gp_br], name='gatherparts_back_low')
        gp_back = SequentialCombination([gp_back_low, gp_bt, ass_b], name='gatherparts_back')

        main_task = HierarchicalTaskHMMSuppRD(root=
                                              ParallelCombination([f_base, gp_back], name='complete'),
                                              name='TaskA',
                                              num_feats_action=self.NUM_FEATS,
                                              feats=self.FEAT,
                                              supp_bhvs=self.SUPP_BHVS,
                                              obj_presence=self.OBJ_PRESENCE,
                                              obj_count_idx=self.OBJ_COUNT_IDX,
                                              main_obj=self.MAIN_OBJ)
