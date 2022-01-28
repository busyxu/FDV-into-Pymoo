# -- coding: utf-8 --
"""
@Project: pymoo
@Time : 2022/1/27 15:25
@Author : Yang xu
@Site :
@File : nsga2.py
@IDE: PyCharm
"""
import copy

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
# from pymoo.core.problem import Problem
from pymoo.util.termination.max_gen import MaximumGenerationTermination as MaxGen
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a  = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


# =========================================================================================================
# Implementation
# =========================================================================================================


class FDVGA(GeneticAlgorithm):

    def __init__(self,
                 rate=0.8,
                 acc=0.4,
                 max_gen=100,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(prob=None, eta=20),
                 survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        rate : {fuzzy evolution rate. Default = 0.8}
        acc : {step acceleration. Default = 0.4}
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         advance_after_initial_infill=True,
                         **kwargs)
        self.rate = rate
        self.acc = acc
        self.max_gen = max_gen
        self.default_termination = MultiObjectiveDefaultTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

    def fdv_operator(self, off):
        # import copy
        off_ = copy.deepcopy(off)
        iter = self.n_gen / self.max_gen
        if iter > self.rate:
            return off
        Xp = off_.get("X")
        total = 1
        S = int(np.sqrt(2*self.rate*total/self.acc))
        # S = floor(sqrt(2 * Rate * Total / Acc))
        # Step(1) = 0，Step(S + 2) is the compensation step
        Step = np.zeros(S + 2,)
        for i in range(S):
            Step[i+1] = (S * (i+1) - (i+1) * (i+1) / 2) * self.acc

        Step[S+1] = self.rate * total # compensation step
        # % step = [0, 0.6, 0.8, 0.8]
        # % % Fuzzy Operation
        # self.problem.xu - self.problem.xl
        R = self.problem.xu - self.problem.xl
        # print(iter)
        for idx in range(S + 1):
            i = idx+1
            if iter > Step[idx] and iter <= Step[idx + 1]:
                part_a = R*10**-i
                part_b = R**-1 * 10**i
                part_c = Xp - self.problem.xl # N*D - 1*D
                gamma_a = part_a * np.floor(part_b * part_c) + self.problem.xl
                gamma_b = part_a * np.ceil(part_b * part_c) + self.problem.xl
                # gamma_a = R * 10**-i * np.floor(R**-1 * 10**i * (Xp - self.problem.xl)) + self.problem.xl
                # gamma_b = R * 10**-i * np.ceil(R**-1 * 10**i * (Xp - self.problem.xl)) + self.problem.xl

                # 与Xp越相似，隶属度越大
                import sys
                mindouble = 1/sys.maxsize
                miu1 = 1 / (Xp - gamma_a + mindouble)
                miu2 = 1 / (gamma_b - Xp + mindouble)
                logical1 = miu1 - miu2 > 0
                logical2 = miu1 - miu2 <= 0
                new_Xp = gamma_a*(logical1) + gamma_b*logical2
                # if self.n_gen==800:
                #     print("stop")

                # print(iter)
                # print(new_Xp)
                return off_.new("X", new_Xp)

        return off

    def _infill(self):  # 重写父类方法

        # do the mating using the current population
        off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self) # infill.py

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        off1 = self.fdv_operator(off)
        return off1


def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding


parse_doc_string(FDVGA.__init__)
