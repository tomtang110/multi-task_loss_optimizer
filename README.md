# multi-task_losses_optimizer
Implement the Pareto Optimizer and pcgrad to make a self-adaptive loss for multi-task

已经实验过了，不会有cuda out of memory情况 

##Pareto optimizer </br>

```python
from Pareto_fn import pareto_fn
w_list = [w1,w2,...]
c_list = [c1,c2,...]
[loss1,loss2,...] = model(inputs)
loss_list = [loss1,loss2,...]
# config is the superparameter for training
new_w_list = pareto_fn(w_list,c_list,config,loss_list)
loss = 0
for i in range(len(w_list)):
    loss += new_w_list[i]*loss_list[i]
model.zero_grad()

loss.backward()
optimizer.step()
```
##pcgrad optimizer </br>

```python
from pcgrad_fn import pcgrad_fn

[loss1,loss2,...] = model(inputs)
loss_list = [loss1,loss2,...]
# config is the superparameter for training

pcgrad_fn(model,loss_list,optimizer)

optimizer.step()
```

## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}

```

```
paper: "A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation". RecSys, 2019, Alibaba
```