import moxel

model = moxel.Model('strin/mnist:latest', where='localhost')
res = model.predict(img=moxel.space.Image.from_file('imgs/digit-2-rgb.png'))
print(res)
