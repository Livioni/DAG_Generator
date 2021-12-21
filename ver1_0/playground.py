from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG

edges,in_degree,out_degree,position = DAGs_generate('default')
# edges,in_degree,out_degree,position = DAGs_generate('random')
plot_DAG(edges,position)
 