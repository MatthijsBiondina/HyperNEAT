from copy import deepcopy
import networkx as nx
from networkx import DiGraph
import numpy as np


"""
    References:
    KO Stanley et al., (2002). Neuroevolution through Augmenting Topologies.
"""
class Evolution:

    def __init__(self,
                 genomes=None,
                 fitness=None,
                 random=None,
                 c1=1.0,
                 c2=1.0,
                 c3=0.4,
                 dt=3.0,
                 prob_add_link=0.1,
                 prob_add_node=0.03,
                 prob_mut_weights=0.8,
                 prob_mut_uniform=0.9,
                 prob_interspecies_mating=0.001,
                 elitism=0.2,
                 weight_range=None,
                 function_set=None
                 ):
        self.genomes      = genomes
        self.fitness      = fitness
        self.random       = random
        self.c1           = c1
        self.c2           = c2
        self.c3           = c3
        self.dt           = dt
        self.prob_add_link = prob_add_link
        self.prob_add_node = prob_add_node
        self.prob_mut_weights = prob_mut_weights
        self.prob_mut_uniform = prob_mut_uniform
        self.prob_interspecies_mating = prob_interspecies_mating
        self.elitism      = elitism
        self.weight_range = weight_range
        self.function_set = function_set

        #local parameters
        self.species = []
        self.species_assignment = {}

        self._init_innovations(genomes=self.genomes)

    """EVOLUTION"""

    """
    Evolve a population of genomes to the next generation, based on their fitness scores
    """
    def evolve(self,fitness_scores=None,genomes=None):
        self.fitness = self._softmax(fitness_scores.tolist())
        self.genomes = genomes
        self.assignSpecies()
        next_gen = []
        N = len(self.genomes)
        #Some individuals are passed to the next generation with only mutation
        for n in range(int(N*self.elitism)):
            next_gen.append( self.random.choice( self.genomes, p=self.fitness ) )
            if self.random.rand() < self.prob_mut_weights:
                next_gen[-1] = self.mutateWeights(parent_genome=next_gen[-1])
            if self.random.rand() < self.prob_add_node:
                next_gen[-1] = self.mutateAddNode(parent_genome=next_gen[-1])
            if self.random.rand() < self.prob_add_link:
                next_gen[-1] = self.mutateAddLink(parent_genome=next_gen[-1])
        #The rest are created through crossover
        while len(next_gen) < len(self.genomes):
            #randomly sample parent 1 based on fitness
            p1_id   = self.random.choice( N, p=self.fitness )
            parent1 = self.genomes[p1_id]
            fitness1 = self.fitness[p1_id]
            if self.random.rand() < self.prob_interspecies_mating:      #if interspecies mating
                p2_id = self.random.choice( N, p=self.fitness )         #sample second parent random as well
                parent2 = self.genomes[p2_id]
                fitness2 = self.fitness[p2_id]
            else:                                                       #otherwise
                specie = self.species[self.species_assignment[ p1_id ]] #sample from species
                p2_id = self.random.choice( len(specie['genomes']), p=specie['fitness'] )
                parent2 = specie['genomes'][p2_id]
                fitness2 = specie['fitness'][p2_id]
            next_gen.append( self.crossover( genome1=parent1,
                                             genome2=parent2,
                                             fitness1=fitness1,
                                             fitness2=fitness2) )
            if self.random.rand() < self.prob_mut_weights:
                next_gen[-1] = self.mutateWeights(parent_genome=next_gen[-1])
            if self.random.rand() < self.prob_add_node:
                next_gen[-1] = self.mutateAddNode(parent_genome=next_gen[-1])
            if self.random.rand() < self.prob_add_link:
                next_gen[-1] = self.mutateAddLink(parent_genome=next_gen[-1])
        return next_gen
                            
                
            
        

    """
    Assign individuals to species for crossover
    """
    def assignSpecies(self):
        assert len(self.genomes) == len(self.fitness)
        #choose representatives for species
        representatives = []
        for specie in self.species:
            if len( specie['genomes'] ) > 0:
                representatives.append( self.random.choice(specie['genomes']))
        self.species = [{'fitness':[],'genomes':[]}] * len(representatives)
        #assign genomes to species
        for idx_in_population,(fitness_score,genome) in enumerate(zip(self.fitness,self.genomes)):
            new_species = True
            #compare each genome to the representative of each species
            for s_id,species_representative in enumerate(representatives):
                if self.delta(genome1=genome,genome2=species_representative) < self.dt:
                    new_species = False
                    self.species[s_id]['fitness'].append(fitness_score)
                    self.species[s_id]['genomes'].append(genome)
                    self.species_assignment[idx_in_population] = s_id
                    break
            if new_species:
                representatives.append(genome)
                self.species_assignment[idx_in_population] = len(self.species)
                self.species.append({'fitness':[fitness_score],'genomes':[genome]})
        # determine shared fitness
        for idx in range(len(self.genomes)):
            self.fitness[idx] = self.fitness[idx] / len( self.species[self.species_assignment[idx]]['fitness'] )
        self.fitness = self._normalize(self.fitness)
        #normalize to use it as a probability distribution
        for specie in self.species:
            specie['fitness'] = self._normalize(specie['fitness'])
                
        
        

    """
    Calculate "the compatibility distance delta between two genomes as a linear combination of
    the number of excess (E) and disjoint (D) genes, as well as the average weight
    differences of matching genes (W), including disabled ones" (Stanley et al., 2002)
    """
    def delta(self,
              genome1=None,
              genome2=None):
        genes1 = set( genome1['links'].keys() )
        genes2 = set( genome2['links'].keys() )
        N = max( len(genes1), len(genes2) )
        if N == 0: #to prevent zero-divisions
            return 0
        #determine excess
        max1 = 0 if len(genes1) == 0 else max(genes1)
        max2 = 0 if len(genes2) == 0 else max(genes2)
        E = sum( 1 for nr in genes1 if nr > max2 ) + sum( 1 for nr in genes2 if nr > max1 )
        #determine disjoint
        D = len( genes1.symmetric_difference(genes2) ) - E
        #determine weigh differences of matching genes
        matching_genes = genes1.intersection(genes2)
        W = 0. if len(matching_genes) == 0 else np.mean( [ abs( genome1['links'][match_nr]['weight'] - genome2['links'][match_nr]['weight'] ) for match_nr in matching_genes ] )
        #calculate delta
        return self.c1*E/N + self.c2*D/N + self.c3*W
        

    """CROSSOVER AND MUTATION"""

    """
    "When crossing over, the genes in both genomes with the same innovation numbers are
    lined up. These genes are called matching genes. Genes that do not match are either
    disjoint or excess, depending on whether they occur within or outside the range of
    the other parent's innovation numbers. They represent structure that is not present
    in the other genome. In composing the offspring, genes are randomly chosen from either
    parent at matching genes, whereas excess or disjoint genes are always included from the
    more fit parent." (Stanley et al., 2002)
    """
    def crossover(self,
                  genome1=None,
                  genome2=None,
                  fitness1=0.,
                  fitness2=0.):
        #Instantiate child genome
        child_genome = {'nodes':[],'links':{}}
        #dominant genome is the genome of the more fit parent
        #recessive genome is the genome of the less fit parent
        dom_genome,rec_genome = (genome1,genome2) if fitness1 > fitness2 else (genome2,genome1)
        # matching genes are chosen at random from either parent,
        # disjoint and excess genes are always chosen from the more fit parent
        for innov_nr in dom_genome['links']:
            if innov_nr in rec_genome['links']:
                child_genome['links'][innov_nr] = dom_genome['links'][innov_nr] if self.random.rand() > 0.5 else rec_genome['links'][innov_nr]
            else:
                child_genome['links'][innov_nr] = dom_genome['links'][innov_nr]
        #Since disjoint and excess genes are always chosen from the more fit parent,
        # the resulting node structure in the child genome is the same as in the parent
        # genome
        child_genome['nodes'] = dom_genome['nodes']
        return child_genome    

    """
    Add a link between two nodes in the genome
    """
    def mutateAddLink(self,
                      parent_genome=None):
        #construct a graph representing the CPPN structure of the genome
        #we need this to guarantee that the CPPN remains acyclic after mutation
        child_genome = deepcopy(parent_genome)
        graph = DiGraph()
        for node in child_genome['nodes']:
            graph.add_node( node['id'] )
        for _,link in child_genome['links'].items():
            graph.add_edge( link['in'], link['out'] )        
        #not all add-link mutations will be legal. We'll try a couple times before giving up
        nr_of_tries = 0
        legal_link_found = False
        while not legal_link_found:
            if nr_of_tries >= 10: #give up
                return child_genome # return without mutation
            in_node  = self.random.choice( child_genome['nodes'] )
            out_node = self.random.choice( child_genome['nodes'] )
            #make a few checks to make sure that this selection is legal
            if in_node == out_node: #can't link to itself
                nr_of_tries += 1
                continue
            elif out_node['type'] == 'Sensor': #can't output into input
                nr_of_tries += 1
                continue
            elif in_node['type'] == 'Output': #can't start link from output
                nr_of_tries += 1
                continue
            elif nx.has_path( graph, out_node['id'], in_node['id'] ): #adding link would make CPPN cyclic
                nr_of_tries += 1
                continue
            else:
                legal_link_found = True
        self.global_innovation_nr += 1
        child_genome['links'][self.global_innovation_nr] = {'in':in_node['id'],
                                                            'out':out_node['id'],
                                                            'weight':self.random.uniform(-self.weight_range,self.weight_range),
                                                            'enabled':True,
                                                            'innov_nr':self.global_innovation_nr}
        return child_genome

    """
    Insert a node in the middle of an existing link
    """
    def mutateAddNode(self,
                      parent_genome=None):
        child_genome = deepcopy(parent_genome)
        #if there are no links, we can't insert a node into one
        if len( child_genome['links'] ) == 0:
            return child_genome
        self.global_innovation_nr += 1
        #add a new node to the genome
        new_node = {'func':self.random.choice(self.function_set),
                    'id':self.global_innovation_nr,
                    'type':'Hidden'}
        child_genome['nodes'].append(new_node)
        #insert the new node into an existing link
        old_link = child_genome['links'][ self.random.choice(np.sort(np.array(list(child_genome['links'].keys())))) ]
        old_link['enabled'] = False
        child_genome['links'][self.global_innovation_nr] = { 'in':old_link['in'],
                                                             'out':new_node['id'],
                                                             'weight':1.,
                                                             'enabled':True,
                                                             'innov_nr':self.global_innovation_nr}
        self.global_innovation_nr += 1
        child_genome['links'][self.global_innovation_nr] = { 'in':new_node['id'],
                                                             'out':old_link['out'],
                                                             'weight':old_link['weight'],
                                                             'enabled':True,
                                                             'innov_nr':self.global_innovation_nr}
        return child_genome

    def mutateWeights(self,
                      parent_genome=None):
        child_genome = deepcopy(parent_genome)
        for _,link in child_genome['links'].items():
            if self.random.rand() < self.prob_mut_uniform: #perturb a little bit
                link['weight'] = min( self.weight_range, max( -self.weight_range, link['weight'] + self.random.uniform( -(self.weight_range/10),(self.weight_range/10))))
            else: #assign new random value
                link['weight'] = self.random.uniform( -self.weight_range, self.weight_range )
        return child_genome
        

    """
    The global innovation number is used to keep track of the order that new genes appear.
    Each time a new gene appears through mutation, the global innovation number is
    incremented and assigned to that gene.
    """
    def _init_innovations(self,
                          genomes=None):
        self.global_innovation_nr = -1
        for genome in genomes:
            for innov_nr in genome['links']:
                self.global_innovation_nr = max(self.global_innovation_nr,innov_nr)
        
    def _normalize(self,lst):
        m = min(lst)
        if m < 0:
            lst = [i-m for i in lst]
        s = sum(lst)
        if s == 0:
            N = len(lst)
            return [1./N] * N
        else:
            return list(map(lambda x: float(x)/s, lst))

    def _softmax(self,x):
        m = min(x)
        if m < 0:
            x = [i-m for i in x]
        x = np.array(x)
        e_x = np.exp(x - np.max(x))
        return (e_x / e_x.sum()).tolist()
