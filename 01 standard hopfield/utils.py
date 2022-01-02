
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_img(img, dim):
    plt.figure(figsize=(dim, dim))
    w_mat = plt.imshow(img, cmap=cm.coolwarm)
    plt.colorbar(w_mat)
    plt.title("Img")
    plt.tight_layout()
    plt.show()