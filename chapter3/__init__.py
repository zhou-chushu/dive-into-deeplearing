from .utils import Accumulator, accuracy, evaluate_accuracy  # classification accuracy and evaluation
from .linreg import synthetic_data # Linear data synthesis

# 相对导入：使用点 (.) 表示当前模块所在的包。这种导入方式通常用于包内的模块之间相互导入。
# 适用场景：当两个模块位于同一个包内时，可以使用相对导入。
# 绝对导入：直接指定模块名称，不使用点 (.)。这种导入方式通常用于跨包的模块导入。
# 适用场景：当两个模块不在同一个包内时，可以使用绝对导入。
# 相对导入：适用于同一包内的模块导入。
# 绝对导入：适用于跨包的模块导入。
# 需要重复调用的用相对调用放在一个文件夹下，重复调用需要作为主文件执行的用绝对调用本文件下的函数用sys改路径调其他文件夹下函数
# __init__.py 使得文件夹对同级代码相当于py文件