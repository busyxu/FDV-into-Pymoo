# FDV-into-Pymoo

## 环境
- windows 11
- python 3.6
- pycharm

## 框架
- pymoo

## 操作
- pip安装第三方框架包（pymoo）。  *pip install pymoo*
- 将开发的算法 fdvga.py复制到pymoo框架中，此处获得fdvga.py。https://github.com/busyxu/FDV-into-Pymoo
- 将下载下来的fdvga.py复制到pymoo根目录\algorithms\moo\
- 成功后的调用方法和pymoo中原有算法的调用相同，具体查看pymoo的官方文档API。 https://pymoo.org/algorithms/moo/nsga2.html

## 测试运行展示
### FDV
**算法FDV在二维DTLZ测试问题上的非支配解集图**

![image](https://user-images.githubusercontent.com/48397805/151512422-b3dd6880-3468-4f20-8c03-77f881e73d96.png)

**算法FDV在三维DTLZ测试问题上的非支配解集图**

![image](https://user-images.githubusercontent.com/48397805/151512688-068e0618-a0ae-4ca6-8921-8f8f25644e1b.png)

### NSGA-II
**算法FDV在二维DTLZ测试问题上的非支配解集图**

![image](https://user-images.githubusercontent.com/48397805/151512872-62251e73-031a-429c-bc86-b053b4c0eb02.png)

**算法FDV在三维DTLZ测试问题上的非支配解集图**

![image](https://user-images.githubusercontent.com/48397805/151512915-c54ae5b4-d99c-4ae5-8ea3-c280960c207d.png)
