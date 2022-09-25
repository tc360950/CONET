
from newick import loads

with open("./tree_newick") as f:
    tree_ =f.read()

tree_=tree_.replace("\"", "").replace("}", ">").replace("{", "<").replace(":", "")
print(tree_)
print(loads(tree_))

print([n.name for n in loads(tree_)[0].descendants])