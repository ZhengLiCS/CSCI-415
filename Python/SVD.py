import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


img = Image.open('datasets/titanic.jpeg')
array = np.array(img)

U, D, V = [], [], []
for channel in np.transpose(array, axes=(2, 0, 1)):
    u, d, v = np.linalg.svd(channel)
    U.append(u)
    D.append(np.diag(d))
    V.append(v)
U = np.stack(U, axis=2)
D = np.stack(D, axis=2)
V = np.stack(V, axis=2)


fig, axes = plt.subplots(2, 3, tight_layout=True)
axes[0, 0].imshow(array)
axes[0, 0].title.set_text("original image")
axes[0, 0].title.set_size(10)
for i, n in enumerate([5, 10, 15, 20, 25]):
    image = np.asarray(np.einsum("ijk,jlk,jmk->imk", U[:, :n, :], D[:n, :n, :], V[:n, :, :]), dtype=np.int32)
    axes[(i + 1) // 3, (i + 1) % 3].imshow(image)
    axes[(i + 1) // 3, (i + 1) % 3].title.set_text("{} Singular Values".format(n))
    axes[(i + 1) // 3, (i + 1) % 3].title.set_size(10)

plt.show()
