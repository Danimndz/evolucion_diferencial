import numpy as np 
import cv2

imgA = cv2.imread(r'C:\Users\wachu\Documents\IIA\Semestre 6\Optimizacion & Metaheuristicas II\P2\proyecto\GTHsprsmall.jpg',1)
auxImg = cv2.imread(r'C:\Users\wachu\Documents\IIA\Semestre 6\Optimizacion & Metaheuristicas II\P2\proyecto\CircleGsprsmall.jpg',1)
numI = 1999
maxImgValue = 100
minRadius = 2
maxRadius = 50
thickness = -1

pathToSave = r"C:\Users\wachu\Documents\IIA\Semestre 6\Optimizacion & Metaheuristicas II\P2\proyecto"


def poblacion_inicial(n):
    pobIn = []
    for i in range(n):
        individuo = []
        center_coords1 = np.random.randint(0,maxImgValue)
        center_coords2 = np.random.randint(0,maxImgValue)
        radius = np.random.randint(minRadius,maxRadius)
        colorR = np.random.randint(0,255)
        colorG= np.random.randint(0,255)
        colorB= np.random.randint(0,255)
        individuo.append(int(center_coords1))
        individuo.append(int(center_coords2))
        individuo.append(int(radius))
        individuo.append(int(colorR))
        individuo.append(int(colorG))
        individuo.append(int(colorB))
        individuo.append(int(thickness))
        pobIn.append(individuo)
    return np.array(pobIn)

def fitnessIndividuo(ind):
    #imgAaux = np.copy(imgA)
    # print("valores: ",int(ind[0]),int(ind[1]),int(ind[2]),int(ind[3]),int(ind[4]),int(ind[5]))
    imInd = np.copy(auxImg)
    imInd = cv2.circle(imInd,(int(ind[0]),int(ind[1])),int(ind[2]),(int(ind[3]),int(ind[4]),int(ind[5])),thickness)
    fit = np.sum( ( imgA.astype("float")-imInd.astype("float") )**2) #Faltaban parentesis
    return fit

def fitness(pob):
    fitnesPob = []
    for i in pob:
        fitnesPob.append(fitnessIndividuo(i))
    return fitnesPob

def buscar_elite(pob,fit):
    indexFit = np.argmin(fit)
    minFit = np.amin(fit)
    return pob[indexFit],minFit

def mutacion(pob):
    pob = pob.astype(float)
    pobMut = []
    for i in range(len(pob)):
        indexX1 = np.random.randint(len(pob))
        indexX2 = np.random.randint(len(pob))
        indexX3 = np.random.randint(len(pob))
        F = np.random.random()*2
        vi = pob[indexX1,:]+F*(pob[indexX2,:]-pob[indexX3,:])
        vi[0] = 0 if vi[0] < 0 else vi[0]
        vi[1] = 0 if vi[1] < 0 else vi[1]
        vi[0] = maxImgValue if vi[0] > maxImgValue else vi[0]
        vi[1] = maxImgValue if vi[1] > maxImgValue else vi[1]

        vi[2] = minRadius if vi[2] < minRadius else vi[2]
        vi[2] = maxRadius if vi[2] > maxRadius else vi[2]

        vi[3] = 0 if vi[3] < 0 else vi[3]
        vi[4] = 0 if vi[4] < 0 else vi[4]
        vi[5] = 0 if vi[5] < 0 else vi[5]
        vi[3] = 255 if vi[3] > 255 else vi[3]
        vi[4] = 255 if vi[4] > 255 else vi[4]
        vi[5] = 255 if vi[5] > 255 else vi[5]

        pobMut.append(vi)
    return np.array(pobMut).astype("int")

def reproduccion(pob,pobM):
    pobRep = []
    for i in range(len(pob)):
        u = np.zeros((7),float)
        for j in range(len(pob[0])):
            r = np.random.random()
            Cr = np.random.random()
            l = np.random.randint(7)
            if j == l: u[j] = pobM[i][l]
            else:
                u[j] = pobM[i][j] if r < Cr else pob[i][j]
        pobRep.append(u)
    return np.array(pobRep).astype("int")

def seleccion(pob,pobRep,fitX,fitU):
    pobSelect = []
    fitSelect = []
    for i in range(len(pob)):
         if fitX[i] < fitU[i]:
             pobSelect.append(pob[i])
             fitSelect.append(fitX[i])
         else:
             pobSelect.append(pobRep[i])
             fitSelect.append(fitU[i])
    return np.array(pobSelect).astype(int),fitSelect

k = 1999
K = 5000
# Ciclo grandote
while k < K: #while fit > 0
    poblacion = poblacion_inicial(200)
    fit = fitness(poblacion)
    elit,elitFit = buscar_elite(poblacion,fit)
    G=50
    g = 0
    print('g:',elit)
    # print('poblacion',poblacion)
    while g < G:
        poblacionM = mutacion(poblacion)
        # print('poblacion mutacion', poblacionM)
        hijos = reproduccion(poblacion,poblacionM)
        # print('poblacion reproduccion', hijos)
        fitP = fitness(poblacion)
        fitR = fitness(hijos)
        poblacion,fit = seleccion(poblacion,hijos,fitP,fitR)
        # print('poblacion seleccion', poblacion)
        if fit[0] < elitFit:
            elitFit = fit[0]
            elit = poblacion[0]
        else:
            poblacion[-1] = elit
            fit[-1] = elitFit
        g+=1
        if g%10 == 0:
            print('g: ',g,'fit: ',elitFit)
            print('elite',elit)
            imgElite = np.copy(auxImg)
            imgElite = cv2.circle(imgElite,(int(elit[0]),int(elit[1])),int(elit[2]),(int(elit[3]),int(elit[4]),int(elit[5])),thickness)
            # cv2.imshow('image',imgElite)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    if k%10 == 0: print("#####################",k,"####################################")
    k+=1
# Agregar el circulo del elite auxImg
    auxImg = cv2.circle(auxImg,(int(elit[0]),int(elit[1])),int(elit[2]),(int(elit[3]),int(elit[4]),int(elit[5])),thickness)
# Guardar la imagen de auxImg    cv2.savefig
    imgName = "myImageSprSmll{}".format(numI)+".jpg"
    cv2.imwrite(pathToSave+imgName,auxImg)
    numI+=1



