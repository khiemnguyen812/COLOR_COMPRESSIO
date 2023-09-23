import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from PIL import Image

# thuật toán Kmean
def k_means(img, k, iter, tol):
    # Ma trận chứa điểm dữ liệu đại diện cho các tâm cụm.
    centroids = img[np.random.choice(img.shape[0], size=k, replace=False)]
    prev_centroids = np.copy(centroids)

    for i in range(iter):
        # Ma trận chứa khoảng cách giữa mỗi điểm dữ liệu và các tâm cụm.
        distance = np.linalg.norm(img - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distance, axis=0)
        # Ma trận chứa giá trị trung bình của các điểm dữ liệu.
        means = []
        for j in range(k):
            means.append(img[labels==j].mean(axis=0))
        centroids = np.array(means)
        # Kiểm tra xem khoảng cách trọng tâm hiện tại và trọng tâm trước đó có nhỏ hơn dung sai đã chỉ định (tol) hay không ?
        if np.linalg.norm(centroids - prev_centroids) < tol:
            break

        prev_centroids = np.copy(centroids)

    return centroids, labels

# Hàm giúp save ảnh
def save_image_as_png(img, file_name):
    try:
        img = Image.fromarray(img)  # Convert the NumPy array to Image object
        img.save(f"{file_name}.png", format="PNG")
        print(f"Image saved as '{file_name}.png'")
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the image: {e}")

# Hàm để mở ảnh from PIL import Image
SUPPORTED_IMAGE_FORMATS = ["PNG", "JPEG", "GIF", "BMP", "ICO", "TIFF", "WEBP"]
def open_image_by_name(image_name):
    try:
        for format in SUPPORTED_IMAGE_FORMATS:
            image_path = f"{image_name}.{format.lower()}"
            try:
                img = Image.open(image_path)
                return img
            except Exception:
                pass
        
        print(f"Error: Unable to open the image '{image_name}' in any supported format.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while opening the image '{image_name}': {e}")
        return None


# Chuyển đổi ảnh thành ma trận
def reshape_img(raw_img):
    matrix = np.array(raw_img)
    matrix_height, matrix_width = matrix.shape[0], matrix.shape[1]
    matrix = matrix.reshape(matrix_height * matrix_width, matrix.shape[2])
    return matrix, matrix_height, matrix_width 

def solve(raw_img,name_img):
    demo,axis=plt.subplots(1,4,figsize=(14,10))
    
    axis[0].set_title(f'Original image: ')
    axis[0].imshow(raw_img)
    # Xử lý ảnh
    n=1
    for k in [3,5,7]:
        img,img_h,img_w=reshape_img(raw_img)
        centroids, labels = k_means(img,k, 100,1e-6)
        for i in range(centroids.shape[0]):
            img[labels==i] =centroids[i]
        
        img=img.astype("uint8")
        img=img.reshape(img_h,img_w,-1)
        #Xuất ảnh
        axis[n].set_title(f'With K = {k}')
        axis[n].imshow(img.copy())
        #Save ảnh
        save_image_as_png(img.copy(),f"{name_img}_{k}")
        n+=1

    plt.tight_layout()

def main():
    
    name_img = input("Enter Name Of Picture: ")
    raw_img = open_image_by_name(name_img)
    solve(raw_img,name_img)
    
    plt.show()
    
main()