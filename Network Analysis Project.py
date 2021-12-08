#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter # Counter 
from itertools import tee, count
import os


# In[2]:


nodes = pd.read_csv("https://raw.githubusercontent.com/Murataydinunimi/Network-Analysis-and-Link-Prediction/main/Dataset/nodes.nodes")
edges = pd.read_csv("https://raw.githubusercontent.com/Murataydinunimi/Network-Analysis-and-Link-Prediction/main/Dataset/edges.edges",header=None)


# In[3]:


edges.columns = ["source","target"]
edges


# In[4]:


n = nodes.columns[0]
nodes = nodes.drop(n,axis=1)
nodes


# In[5]:



len(np.unique(nodes.new_id)),len(np.unique(edges)) 
# number of unique values in edges should be equal to the number of unique values of nodes.
#we have nodes more in the edges dataset that do not match the node dataset.


# In[6]:



not_seen = []

for ids in list(np.unique(edges)): # get the nodes in the edge dataset that do not match the node dataset
    if ids not in list(np.unique(nodes.new_id)):
        not_seen.append(ids)
        
        
for value in not_seen: # remove the unmatched values
    index_firstcol = edges[edges.iloc[:,0] == value].index.to_list()
    index_secondcol = edges[edges.iloc[:,1]== value].index.to_list()
    edges = edges.drop(index_firstcol,axis=0)
    edges = edges.drop(index_secondcol,axis=0)


# In[7]:


len(np.unique(nodes.new_id)),len(np.unique(edges)) 


# In[8]:


nodes.name[nodes.name.duplicated()] # we have 49 duplicate values meaning that some Tv-shows have more than one page.
#to adress this issue, we will add 1,2,3... to each duplicated value.


# In[9]:


def unique_names(names):
    
    duplicate = [k for k,v in Counter(names).items() if v>1] # so we have: ['name', 'zip']
    max_duplicate =max([v for k,v in Counter(nodes.name).items() if v>1])
    suff_generator = dict(zip(duplicate, tee((f'_{x}' for x in range(1, max_duplicate+1)), len(duplicate))))  
    #for each duplicated name, generate an iterable ranges to max duplicated value.
    for idx,name in enumerate(names):
        try:
            suffix = str(next(suff_generator[name]))
        except KeyError:
            # name is already unique.
            continue
        else:
            names[idx] += suffix


# In[10]:


node_names = list(nodes.name)
unique_names(node_names)


# In[11]:


len(node_names)


# In[12]:


n = nodes.columns[0]

nodes.drop(n,axis=1,inplace=True)

nodes[n] = node_names

nodes
        


# In[13]:


len(np.unique(nodes.name))


# In[14]:


list(zip(nodes["name"],nodes["new_id"]))


# In[15]:


nodes_dict = {}
for (n, id) in zip(nodes["name"],nodes["new_id"]):
 # print(n, "+", id)
  nodes_dict[id] = n


# In[16]:


nodes_dict


# **NETWORK BUILDING**
# 

# In[17]:


G = nx.Graph()

for e in edges.values.tolist():
    G.add_edge(e[0],e[1])


# In[18]:


len(G.nodes())


# In[19]:


H = nx.relabel_nodes(G, nodes_dict)
len(sorted(H))

G = H


# In[20]:


print('Number of nodes: {} - Number of links:{}'.format(G.order(),G.size()))


# In[23]:



import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(15,15))
nx.draw_networkx_nodes(G, pos, node_size=10, label = node_names)
nx.draw_networkx_edges(G, pos, alpha=0.3,width=0.3,)
plt.show()


# In[24]:


pos = nx.spring_layout(G)
plt.figure(figsize=(15,15))

nx.draw_networkx_nodes(G, pos, node_size=15, label = node_names)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G,pos,font_size=3,font_color='r')
plt.show()


# In[21]:


degree_df = pd.DataFrame.from_dict(dict(G.degree()),orient="index",columns=["degree"]).reset_index() # degree of each node
degree_df.columns = ["name","degree"]
degree_df =  nodes.merge(degree_df, on ="name")
degree_df


# In[22]:


degrees = list(dict(G.degree()).values())

percentile_99 = np.percentile(degrees, 99 ) # we consider hubs greater than 99 percentile. # 64.19
hubs = [key for key,value in dict(G.degree()).items() if value >= percentile_99] 


# In[23]:


hubs


# In[24]:


nx.set_node_attributes(G, 0, name="is_hub")


# In[25]:



#for gephi visualization

for node,attribute in G.nodes(data=True):
    if node in hubs:
        G.nodes[node]["is_hub"]=1


# In[26]:


number_of_hubs =0
for node,attribute in G.nodes(data=True):
    if G.nodes[node]["is_hub"] == 1:
        number_of_hubs += 1
    
print(number_of_hubs,len(hubs))


# In[27]:


node_names = list(degree_df.name)

colors = []

for i in range(0, len(node_names)):
    if not (node_names[i] in hubs) :
        colors.append('#86b5fb') # non-hub
        
    else: 
        
        colors.append('#93e685') #hub


# In[28]:


from pyvis.network import Network

net = Network(height=400, width=600, bgcolor='#222222', font_color='white', notebook =True)

# set the physics layout of the network
net.force_atlas_2based(gravity=-300,spring_length=300)


# In[29]:


net.add_nodes(degree_df['new_id'].tolist(), size =degree_df['degree'].tolist(),
                         title=degree_df['name'].tolist(),
                         label=degree_df['name'].tolist(),
                          color=colors)


# In[30]:


for index in range(len(edges['source'])):
    try:
        src = int(edges['source'][index])
        dst = int(edges['target'][index])
    except KeyError:
        #the index does not exist because before we deleted some edges where the deletion was done based on index.
        #since we use a range function above, we miss some of those edges.
        #actually, instead of try except block we could simply loop over the unique indices of edges and solve the problem.
        #but I wanted to do it this way :)
        continue
    else:
        
        net.add_edge(src,dst)


# In[38]:


net.options.nodes ={"font" : {
          "size" : 50,
          "color" : '#ffffff'
      }}


# In[ ]:


net.show("companies.html")


# In[28]:


G.nodes["Queen of the South"]


# **DEGRE ANALYSIS**

# In[43]:


density = nx.density(G)

print("density:{}".format(density))


# In[59]:


degree = list(dict(G.degree()).values())


# In[45]:


print("Standard deviation : {}".format(np.std(degree)))
print("Mean: {}".format(np.mean(degree)))
print("Median : {}".format(np.median(degree)))
print("Min_degree: {}".format(np.min(degree)))
print("Max_degree : {}".format(np.max(degree)))

print("Assortativity: " + str(nx.degree_assortativity_coefficient(G)))


# In[63]:


import collections

n_degrees = sorted([d for n,d in G.degree()],reverse=True) # degrees of each node
frequency_degrees = collections.Counter(n_degrees) # groupping by degree numbers and count each realization.
degree_1, counts =zip(*frequency_degrees.items())

plt.figure(figsize=(15,10))

plt.bar(degree_1, counts, width=0.8, color="b")
plt.title("Degree Histogram")
plt.ylabel("Nodes")
plt.xlabel("Degree")


# In[64]:


plt.figure() # you need to first do 'import pylabas plt'plt.grid(True)
plt.figure(figsize=(15,10))
plt.plot(degree_1,counts,"ro-") # degree
plt.legend(['Degree'])
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Network Companies')
plt.show()


# In[65]:


plt.figure() # you need to first do 'import pylabas plt'plt.grid(True)
plt.figure(figsize=(15,10))
plt.loglog(degree_1,counts,"ro-") # degree
plt.legend(['Degree'])
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Log Network Companies')
plt.show()


# **EMPRICAL CUMULATIVE DISTRIBUTION FUNCTION**

# In[66]:


cdf = ECDF(degree)
x = np.unique(degree)
y = cdf(x)
fig_cdf = plt.figure(figsize=(15,10))
axes = fig_cdf.gca()

axes.plot(x,y,marker='o',ms=6,linestyle ="--")
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECDF companies', size = 20)


# In[67]:


# ECDF loglog scale
cdf = ECDF(degree)
x = np.unique(degree)
y = cdf(x)
fig_cdf = plt.figure(figsize=(15,10))
axes = fig_cdf.gca()
axes.loglog(x,y,marker='o',ms=8, linestyle='--')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECDF companies', size = 20)


# In[68]:


# ECCDF
cdf = ECDF(degree)
x = np.unique(degree)
y = cdf(x)
fig_cdf = plt.figure(figsize=(15,10))
axes = fig_cdf.gca()
axes.loglog(x,1-y,marker='o',ms=8, linestyle='--')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECCDF companies', size = 20)


# **RANDOM GRAPH**

# In[43]:


p = density
random_graph = nx.fast_gnp_random_graph(G.order(),p)
print('Number of nodes: {}'.format(random_graph.order()))
print('Number of links: {}'.format(random_graph.size()))


# In[41]:


random_degree = list(dict(random_graph.degree()).values())
print('Random Net Standard deviation: {}'.format(np.std(random_degree)))
print('Random Net Mean: {}'.format(np.mean(random_degree)))
print('Random Net Median: {}'.format(np.median(random_degree)))
print('Random Net Min: {}'.format(np.min(random_degree)))
print('Random Net Max: {}'.format(np.max(random_degree)))
print("Random Net density :{}".format(nx.density(random_graph)))


# In[44]:



local_clustering=nx.clustering(random_graph)
list_local_clustering=list(local_clustering.values())
print('Mean local clustering: {}'.format(np.mean(list_local_clustering)))


# In[45]:


nx.transitivity(random_graph)


# In[40]:


cdf = ECDF(degree)
x = np.unique(degree)
y = cdf(x)

cdf_random = ECDF(random_degree)
x_random = np.unique(random_degree)
y_random = cdf_random(x_random)

fig_cdf_fb = plt.figure(figsize=(15,10))
axes = fig_cdf_fb.gca()
axes.set_xscale('log')
axes.set_yscale('log')
axes.loglog(x,1-y,marker='o',ms=8, linestyle='--')
axes.loglog(x_random,1-y_random,marker='+',ms=10, linestyle='--')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('Random Graph ECDF vs Real Network ECDF', size = 20)


# **BRIDGES**

# In[41]:


nx.has_bridges(G)


# In[42]:


len([ br for br in nx.bridges(G, root=None)])


# In[43]:


nx.set_edge_attributes(G, 0, name="is_bridge")


# In[44]:


for br in nx.bridges(G, root=None):
    print("edge (src,target):", br)
    break


# In[45]:


for br in nx.bridges(G, root=None):
    #print("edge (src,target):", br)
    src,target = br
    if G.has_edge(src,target):
        G[src][target]['is_bridge'] = 1  # if g has such a source and target, set it to 1.
    
    if  G.has_edge(target,src):
        G[target][src]['is_bridge'] = 1  # same as before, it might happen that src target does not exist but target src exist.


# In[46]:


dict(G.edges())['NBC Nightly News with Lester Holt', 'Bare Feet with Mickela Mallozzi']


# **LOCAL BRIDGES**

# In[47]:


nx.set_edge_attributes(G, 0, name="is_local_bridge")


# In[48]:


for br in nx.local_bridges(G, with_span=False, weight=None):
    #print("edge (src,target, span):", br)
    src, target = br
    
    if G.has_edge(src,target):
        G[src][target]['is_local_bridge'] = 1 
    
    if  G.has_edge(target,src):
        G[target][src]['is_local_bridge'] = 1 


# In[49]:


dict(G.edges())['Dancing With The Stars. Taniec z Gwiazdami',
  'Twoja twarz brzmi znajomo']


# **CONNECTIVITY**

# In[50]:


print(list(nx.isolates(G)))

print(nx.is_connected(G))
print(nx.number_connected_components(G))


# **CLUSTERS**

# In[51]:


clusters = nx.average_clustering(G)
clusters


# In[52]:


nx.transitivity(G)


# In[54]:



local_clustering=nx.clustering(G)
list_local_clustering=list(local_clustering.values())
print('Mean local clustering: {}'.format(np.mean(list_local_clustering)))


# In[55]:


LCC = sorted(local_clustering.items(), key=lambda item: item[1], reverse= False)
LCC


# In[56]:


nx.set_node_attributes(G, 0, name="LCC")
for node,attribute in G.nodes(data=True):
    G.nodes[node]["LCC"] = dict(LCC)[node]


# In[57]:


print("TOTAL number of triangles in the graph: ", sum(list(nx.triangles(G).values())))


# **CENTRALITY**

# In[58]:


deg_centr = nx.degree_centrality(G)
sort_orders = sorted(deg_centr.items(), key=lambda x: x[1], reverse=True)

print("10 most important nodes for Degree Centrality:")
for i in range(10):
    print(sort_orders[i])


# In[59]:


#gephi

nx.set_node_attributes(G, 0, name="deg_centr")
for node,attribute in G.nodes(data=True):
    G.nodes[node]["deg_centr"] = dict(sort_orders)[node]


# In[60]:


betweennesCentrality = nx.betweenness_centrality(G)
sort_orders = sorted(betweennesCentrality.items(), key=lambda x: x[1], reverse=True)
print("10 most important nodes for Betweennes Centrality:")
for i in range(10):
    print(sort_orders[i])


# In[61]:


nx.set_node_attributes(G, 0, name="b_centrality")

for node,attribute in G.nodes(data=True):
    G.nodes[node]["b_centrality"] = dict(sort_orders)[node]


# In[62]:


G.nodes["The Voice Global"]


# In[63]:


eigen = nx.eigenvector_centrality(G)
sort_orders = sorted(eigen.items(), key=lambda x: x[1], reverse=True)
print("10 most important nodes for Eigenvector Centrality:")
for i in range(10):
      print(sort_orders[i])


# In[64]:


nx.set_node_attributes(G, 0, name="eigen_centrality")
for node,attribute in G.nodes(data=True):
    G.nodes[node]["eigen_centrality"] = dict(sort_orders)[node]


# In[65]:


G.nodes["New Girl"]


# In[66]:


pagerank = nx.pagerank(G)
sort_orders = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
print("10 most important nodes for Page Rank:")
for i in range(10):
    print(sort_orders[i])


# In[67]:


nx.set_node_attributes(G, 0, name="Page rank")
for node,attribute in G.nodes(data=True):
    G.nodes[node]["Page rank"] = dict(sort_orders)[node]


# In[68]:


nx.diameter(G, e=None, usebounds=False)


# **COMMUNITIES**

# In[69]:


import networkx.algorithms.community as nx_comm

import community.community_louvain as community_louvain

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

list_community_sets_greedy = list(nx_comm.greedy_modularity_communities(G))
print(list_community_sets_greedy[0:20])


# In[70]:


partition_greedy = {}
for i, comm in enumerate(list_community_sets_greedy):
    print("Community:", i)
    print("Number of elems",len(comm))
    for n in comm:
        partition_greedy[n]=i


# In[71]:


print(list(partition_greedy.items())[0:20]) #Wicked city belongs to the community 0 etc.


# In[72]:


nx.set_node_attributes(G, partition_greedy, "community_nx_greedy")
for node,attribute in G.nodes(data=True):
    G.nodes[node]["community_nx_greedy"] = partition_greedy[node]


# In[73]:


G.nodes["Driven MBC"]


# In[74]:


def draw_graph_with_communities(G,partition):
    # draw the graph
    pos = pos=nx.random_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.figure(figsize=(15,10))
    plt.show()
    


# In[75]:


draw_graph_with_communities(G,partition_greedy)


# In[76]:


# draw the graph
pos=nx.fruchterman_reingold_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('tab20', max(partition_greedy.values()) + 1)
plt.figure(figsize=(15,15))

nx.draw_networkx_nodes(G, pos, partition_greedy.keys(), node_size=20,
                       cmap=cmap, node_color=list(partition_greedy.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)

plt.show()


# **LOUVAIN ALGORITHM**

# In[77]:


partition_library = community_louvain.best_partition(G)
print(list(partition_library.items())[0:20])


# In[78]:


nx.set_node_attributes(G, partition_library, "community_louvain")


# In[79]:


G.nodes["Twoja twarz brzmi znajomo"]


# **LPA ALGORITHM**

# In[80]:


import networkx.algorithms.community as nx_comm


# In[81]:


list_community_sets_lpa = list(nx_comm.label_propagation_communities(G))
print(list_community_sets_lpa)


# In[82]:


partition_lpa = {}
for i, comm in enumerate(list_community_sets_lpa):
    print("Community:", i)
    print(i,comm)
    for n in comm:
        partition_lpa[n]=i


# In[83]:


print(partition_lpa)


# In[84]:


nx.set_node_attributes(G, partition_lpa, "community_lpa")


# In[85]:


G.nodes["Le Québec, une histoire de famille"]


# In[86]:


comms = set(partition_library.values())
comms


# In[87]:


list_community_sets_library = [set() for i in range(len(comms)) ]


# In[88]:


for n, comm in partition_library.items():
    list_community_sets_library[comm].add(n)

list_community_sets_library[0:2]


# In[89]:


comms = set(partition_lpa.values())
comms


# In[90]:


list_community_sets_lpa = [ set() for i in range(len(comms)) ]


# In[91]:


for n, comm in partition_lpa.items():
    list_community_sets_lpa [comm].add(n)

list_community_sets_lpa 


# In[92]:


method_names = ["Greedy","Louvain library","LPA"]
for i,my_list in enumerate([list_community_sets_greedy,  list_community_sets_library, list_community_sets_lpa ]):
    
    print(method_names[i])
    print()
    
    #print("Coverage")
    print("Coverage", nx_comm.coverage(G, my_list))
    #print("Modularity")
    print("Modularity", nx_comm.modularity(G, my_list, weight='weight'))
    #print("Performance")
    print("Performance", nx_comm.performance(G, my_list))
    
    print("------------------------------------------------------------")
    print()
    


# # Size distribution of communities
# 

# In[93]:


list_community_sets_library


# In[94]:


pairs = []
for i, nodes in enumerate(list_community_sets_library):
    print(i,len(nodes))
    comm_size = (i,len(nodes))
    pairs.append(comm_size)


# In[95]:


pairs = []
for i, nodes in enumerate(list_community_sets_library):
    print(i,len(nodes))
    comm_size = (i,len(nodes))
    pairs.append(comm_size)


# In[96]:


community_index = []
number_of_nodes = []

for comm, n_nodes in pairs:
    community_index.append(str(comm))
    number_of_nodes.append(n_nodes)    


# In[97]:


plt.figure(figsize=(10,8))

plt.bar(community_index,number_of_nodes)
plt.xlabel("Community")
plt.ylabel("Number of nodes")


# # Centrality in communities
# 

# In[98]:


list_community_sets_library # if we want to find the influencers for example, normal centrality measure will give us the 
# global influencers/hubs in the network. While in this way, we can find them in the communities.


# In[99]:


for comm in list_community_sets_library:
    subgraph = G.subgraph(comm)
    print(subgraph.order())


# In[100]:


centr_comm = {}


# In[101]:


for comm in list_community_sets_library:
    subgraph = G.subgraph(comm)
    print(subgraph.order())
    print(nx.degree_centrality(subgraph))
    print("---")


# In[102]:


for comm in list_community_sets_library:
    subgraph = G.subgraph(comm)
    print(subgraph.order())
    print(nx.degree_centrality(subgraph))
    
    node_degrees  = nx.degree_centrality(subgraph)
    for n,d in node_degrees.items():
        centr_comm[n] = d


# In[103]:


centr_comm


# In[104]:


nx.set_node_attributes(G, centr_comm, "centr_comm")


# In[105]:


nx.write_graphml(G,path="pages.graphml")


# **LINK PREDICTION**

# In[106]:


import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[109]:


sources = np.unique(edges["source"]).tolist()
targets = np.unique(edges["target"]).tolist()
dupp_nodes = sources + targets
nodes = (list(dict.fromkeys(dupp_nodes)))
len(nodes)


# In[110]:


new_net = nx.from_pandas_edgelist(edges,"source","target",create_using = nx.Graph())
adj_G = nx.to_numpy_matrix(new_net, nodelist = nodes)


# In[111]:


nodes


# In[112]:


nx.shortest_path_length(new_net,0,14)


# In[113]:


unconnected = []
indices = []
n,d = np.shape(adj_G)

for i in tqdm(range(n)):
    for j in range(i+1,d):
        try:
            path_len = nx.shortest_path_length(new_net,i,j)
        except nx.NodeNotFound:
            break
        else:
            if path_len <=2:    
                if adj_G[i,j] == 0:
                    unconnected.append([nodes[i],nodes[j]])
                    indices.append([i,j])


# In[114]:


len(unconnected) # all unconnected pairs having a path length equal or lower than 2.
#these nodes will be our negative examples in the training of the model.


# In[115]:


unconnected


# In[116]:


nx.shortest_path_length(new_net,0,180)


# In[117]:


node_1_unlinked = [i[0] for i in unconnected]
node_2_unlinked = [i[1] for i in unconnected]

data = pd.DataFrame({'source':node_1_unlinked, 
                     'target':node_2_unlinked})

# add target variable 'link'
data['link'] = 0
data


# In[119]:


initial_node_count = len(new_net.nodes)

edges_temp = edges.copy()

# empty list to store removable links
omissible_links_index = []

for i in tqdm(edges.index.values):
    
      # remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(edges_temp.drop(index = i), "source", "target", create_using=nx.Graph())
  
  # check there is no spliting of graph and number of nodes is same
    if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
        omissible_links_index.append(i)
        edges_temp = edges_temp.drop(index = i)


# In[120]:


len(omissible_links_index) # we can drop all of them and they will be our positive values.


# In[121]:


# we know append those values as positive samples to our unconnected data.


# In[122]:


# create dataframe of removable edges
edges_omis = edges.loc[omissible_links_index]

# add the target variable 'link'
edges_omis['link'] = 1

data = data.append(edges_omis[['source', 'target', 'link']], ignore_index=True)


# In[123]:


data['link'].value_counts() 


# In[124]:


data


# In[125]:


# drop removable edges
edges_partial = edges.drop(index=edges_omis.index.values)


# In[126]:


# build graph

G_data = nx.from_pandas_edgelist(edges_partial, "source", "target", create_using=nx.Graph())


# In[127]:


from node2vec import Node2Vec

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=7, min_count=1)


# In[128]:


x = [(n2w_model.wv[str(i)]+n2w_model.wv[str(j)]) for i,j in zip(data['source'], data['target'])]


# In[129]:


xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)


# In[130]:


lr = LogisticRegression(class_weight="balanced")

lr.fit(xtrain, ytrain)


# In[131]:


predictions = lr.predict_proba(xtest)
roc_auc_score(ytest, predictions[:,1])


# In[132]:


from sklearn import metrics
y_pred = np.argmax(predictions, axis=1)

cnf_matrix = metrics.confusion_matrix(ytest, y_pred)
cnf_matrix


# In[133]:


from sklearn.metrics import roc_curve
from sklearn import metrics

fpr1, tpr1, thresh1 = roc_curve(ytest, predictions[:,1], pos_label=1)
random_probs = [0 for i in range(len(ytest))]
p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
auc_score1 = roc_auc_score(ytest, predictions[:,1])

import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[134]:


import scikitplot as skplt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#figure(figsize=(15, 20), dpi=80)
skplt.metrics.plot_roc(ytest, predictions)
plt.gcf().set_size_inches(20, 10)
plt.show()


# In[135]:


import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}

# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)


# In[ ]:




