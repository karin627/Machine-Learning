import numpy as np
import matplotlib.pyplot as plt

epoch = 10

prediction = []
x = np.zeros((0, 2))

# label = "PartI   Generative Model"
# for y in open('./output_part1/generative/plot/prediction.txt', 'r'):
#     prediction.append(float(y))
    
# label = "PartI   Discriminative Model   EPOCH="+str(epoch)
# for y in open('./output_part1/discriminative/plot/prediction_e'+str(epoch)+'.txt', 'r'):
#     prediction.append(float(y))
    
# label = "PartII   Generative Model"
# for y in open('./output_part2/generative/plot/prediction.txt', 'r'):
#     prediction.append(float(y))

label = "PartII   Discriminative Model   EPOCH="+str(epoch)
for y in open('./output_part2/discriminative/plot/prediction_e'+str(epoch)+'.txt', 'r'):
    prediction.append(float(y))
    
for i in range(101):
    for j in range(101):
        x = np.vstack((x, [i,j]))
        
for cls in range(4):
    points = np.squeeze(np.take(x, np.where(np.array(prediction) == cls), axis=0))
    if cls == 0:
        color = "red"
    elif cls == 1:
        color = "blue"
    elif cls == 2:
        color = "green"
    elif cls == 3:
        color = "yellow"
    plt.scatter(points[:, 0], points[:, 1], color = color)

plt.xlabel('x1 : offensive')
plt.ylabel('x2 : defensive')
plt.legend()
plt.title(label)
plt.show()