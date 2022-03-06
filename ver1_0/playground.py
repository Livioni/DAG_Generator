import DAGs_Generator as DG

edges,in_degree,out_degree,position = DG.DAGs_generate('default')
# edges,in_degree,out_degree,position = DAGs_generate('random')
DG.plot_DAG(edges,position)
print(DG.admatrix(edges,10))
 