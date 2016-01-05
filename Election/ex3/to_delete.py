from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)



# with open("iris.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf(r"c:\temp\iris.pdf")

#
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
#
# __author__ = 'ORI'

#
#
# from IPython.display import Image
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data,
#
#
#
#
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
