import cv2
import numpy as np
def cvshow(a,img):
    cv2.imshow(a,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def absb(a,b):
    c = a-b
    if c<0 :
        c= -c
    return c


img = cv2.imread("C:\\Users\\txjoe\\Desktop\\42.jpg")
img1 = img.copy()
#cvshow('1',img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cvshow('1',img)
img = cv2.GaussianBlur(img, (3, 3), 1)
ret,img = cv2.threshold(img,100,230,cv2.THRESH_BINARY_INV)
#cvshow('1',img)
kernel = np.ones((3, 3))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
ret,img = cv2.threshold(img,220,230,cv2.THRESH_BINARY_INV)
#cvshow('1',img)
#img = cv2.Canny(img, 100, 150)
#轮廓检测
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
draw_img = img1.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)
#cvshow('1',res)


# cnt = contours[2]
# epsilon = 0.05 * cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# draw1_img = img1.copy()
# res = cv2.drawContours(draw_img, [approx], -1, (0, 255, 0), 2)
# cvshow('1',res)

# x, y, w, h = cv2.boundingRect(cnt)  # 对边界计算外接矩形
# img = cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cvshow('1',img)

x1 = []
y1 = []
x2 = []
y2 = []
w1 = []
h1 = []
for c in contours:
    # 找到边界坐标
    x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
    print(x, y, w, h)
    # 因为这里面包含了，图像本身那个最大的框，所以用了if，来剔除那个图像本身的值。
    if x != 0 and y != 0 and w != img.shape[1] and h != img.shape[0] and h/w > 1.5:
        img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x1.append(x)
        y1.append(y)
        w1.append(w)
        h1.append(h)
        x2.append(x + w)
        y2.append(y + h)
        #cvshow('2',img1)

# x11 = min(x1)
# y11 = min(y1)
# x22 = max(x2)
# y22 = max(y2)
#print(x1[0])
#差值c

# x3 = (x2[0]-x1[1])/2
# y3 = (y2[0]-y1[1])/2

img2 = cv2.rectangle(img1, (x2[1], y2[1]), (x2[1]+x1[0]-x2[1],y2[1]+y1[0]-y2[1]), (0, 255, 0), 2)
#cvshow('img1',img2)


for x in x1:
    print(x)
    a=0
    x11 = x2[a]
    y11 = y2[a]

    for y in y1:
        print(y)
        b=0
        x22 = x2[b]
        y22 = y2[b]
        print(absb(x22 , x11))
        print(absb(y11 , y22))
        if 80<absb(x11,x22)<160 and 0< absb(y11,y22) <40 :
            print(absb(x11,x22))
            print(absb(y11,y22))
            if x11>x22:
                t=x11
                x11=x22
                x22=t

            if y11>y22:
                p = y11
                y11 = y22
                y22= p

            img1 = cv2.rectangle(img1, (x11, y11), (x11+ absb(x11,x22), y11+absb(y11,y22)), (0, 255, 0), 2)
            cvshow('img1', img1)
            b=b+1
            break
        else:
            b=b+1

    a = a + 1
