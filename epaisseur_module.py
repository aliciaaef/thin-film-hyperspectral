# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:16:59 2021

@author: pasquet
"""

from scipy.ndimage import median_filter
from arsf_envi_reader import numpy_bin_reader
from arsf_envi_reader import envi_header
from optifik.minmax import thickness_from_minmax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import copy
from tqdm import tqdm
from scipy.signal import savgol_filter

#for line in in_data: #line = bands * samples

def get_datacube(file_bil):
    '''
    file_bil est le datacube produit par le logiciel spectronon
    produit :
        - un datacube (out_datacube) à 3D contenant les valeurs d'intensité
        dans les trois directions temps, espace, lambdas
        - un vecteur des instants (v_temps) en secondes
        - un vecteur des longueurs d'onde (v_lambdas)
    '''
    
    #lecture .hdr et récupération informations
    hdr = envi_header.read_hdr_file(file_bil[:-4]+'.hdr', keep_case=False) #type dictionnaire
    
    n_temps = int(hdr['lines'])
    n_espace = int(hdr['samples'])
    n_lambdas = int(hdr['bands'])
    framerate = float(hdr['framerate'])
    v_lambdas = np.array(list((map(float,list(hdr['wavelength'].split(','))))))

    v_temps = np.arange(0,n_temps)/framerate #secondes
    v_espace = np.arange(n_espace)
    
    #lecture .bil et récupération informations
    in_datacube = numpy_bin_reader.BilReader(file_bil)
    
    #création et remplissage du datacube
    out_datacube = np.zeros([n_temps,n_espace,n_lambdas],dtype=np.float32) #temps (line), espace (sample), lambdas (band)
    
    i_temps=0
    
    for line in in_datacube: #parcourt le temps
        for i_espace in range(n_espace): #parcourt l'espace
            
            intensites = line[:,i_espace] #récupère les intensités pour un point
                                    #en temps et en espace mais pour tous lambdas
            out_datacube[i_temps,i_espace,:] = intensites
        
        i_temps+=1
        
    return(out_datacube,n_temps,n_espace,n_lambdas,v_temps,v_espace,v_lambdas)

def get_thickness_spectronon(file_bil,file_csv):
    '''
    file_bil est le datacube obtenu dans spectronon avec le plugin thinfilm thickness
    la fonction recupere les informations de ce datacube et
    trace l'epaisseur en temps et en espace dans un plot 3D'
    '''
    
    hdr = envi_header.read_hdr_file(file_bil+'.hdr', keep_case=False)
    n_temps = int(hdr['lines'])
    n_espace = int(hdr['samples'])
    framerate = float(hdr['framerate'])
    v_temps = np.arange(0,n_temps)/framerate #secondes
    v_espace = np.array([i for i in range(0,n_espace)])

    #lecture .bil et récupération informations
    in_datacube = numpy_bin_reader.BilReader(file_bil)
    
    #création et remplissage du datacube
    epaisseurs = np.zeros([n_temps,n_espace]) #temps (line), espace (sample), lambdas (band)
    
    i_temps=0

    for line in in_datacube: #parcourt le temps

        epaisseurs[i_temps,:] = line[0]*1000
        
        i_temps+=1
    
    save_csv(epaisseurs,v_temps,v_espace,file_csv)
        
    return(epaisseurs, v_temps, v_espace)

def infos_from_csv(file):
    '''
    récupère les informations d'un csv contenant les épaisseurs calculées
    epaisseurs[temps,espace]
    '''
    
    epaisseurs = []
    with open(file, newline='') as fichier:
        data = csv.reader(fichier,quoting = csv.QUOTE_NONNUMERIC,delimiter=' ',)
        for i in data:
            epaisseurs.append(i)
            
    v_temps  = np.array(epaisseurs[len(epaisseurs)-2])
    v_espace = np.array(epaisseurs[len(epaisseurs)-1])
    epaisseurs = epaisseurs[:len(epaisseurs)-2]
    epaisseurs = np.array(epaisseurs)
    
    #calcul de la proportion de spectres gardés
    prop_spectres_gardes = len(epaisseurs[np.logical_not(np.isnan(epaisseurs))])/np.size(epaisseurs)
    print('proportion de spectres gardés =',prop_spectres_gardes)
    
    return(epaisseurs,v_temps,v_espace,prop_spectres_gardes)

def surf3D(v_temps,v_espace,epaisseurs,title):
    [x,y] = np.meshgrid(v_espace,v_temps)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x,y,epaisseurs)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('t (s)')
    ax.set_zlabel('h (nm)')
    # plt.title(title)
    ax.invert_xaxis()
    
def thickness_map_pcolor(v_temps,v_espace,epaisseurs,title):
    # [x,y] = np.meshgrid(v_espace,v_temps)
    fig,ax = plt.subplots()
    plt.pcolormesh(v_espace,v_temps,epaisseurs/1000,shading='auto')
    # fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # plt.imshow(epaisseurs/1000,interpolation='none') #utilise pour enregistrer en svg, commenter ax.invert_yaxis()
    plt.colorbar(label='h ($\mu$m)')
    plt.xlabel('x (px)')
    plt.ylabel('t (h)')
    # plt.title(title)
    ax.invert_yaxis()
    # ax.axis('image')   
    # ax.set_box_aspect(2)
    # ax.set_aspect('auto')
    # ax.axis('off')
    # ax.axis('tight')
    # cb = plt.colorbar() 
    # cb.remove() 
    # ax.draw()
     
def post_process(epaisseurs,v_temps,v_espace,window,epaisseur_max,diff,window_med):
    
    #surf 3D pas lissée
    surf3D(v_temps,v_espace,epaisseurs,'thickness map')
    thickness_map_pcolor(v_temps,v_espace,epaisseurs,   'thickness map')
    
    #surf 3D filtrée sur les nan
    epaisseurs1 = filtre_median2D_sur_nan(epaisseurs)
    surf3D(v_temps,v_espace,epaisseurs1,'thickness map filtrée sur les nan')
    thickness_map_pcolor(v_temps,v_espace,epaisseurs1,   'thickness map filtrée sur les nan')

    #surf 3D lissée
    epaisseurs2 = median_filter(epaisseurs1,(window,window))
    surf3D(v_temps,v_espace,epaisseurs2,'thickness map lissée, window = ' + str(window))
    thickness_map_pcolor(v_temps,v_espace,epaisseurs2,   'thickness map lissée, window = ' + str(window))

    #enlever épaisseurs trop grandes
    print()
    print('calcul filtre epaisseur max')
    epaisseurs3 = filtre_epaisseur_max_2D_1point(epaisseurs1,epaisseur_max)
    print('filtre epaisseur max calculé')
    surf3D(v_temps,v_espace,epaisseurs3,'thickness map filtrée max ' + str(epaisseur_max))
    thickness_map_pcolor(v_temps,v_espace,epaisseurs3,   'thickness map filtrée max ' + str(epaisseur_max))
    
    print()
    print('calcul filtre différence voisins')
    epaisseurs4 = filtre_epaisseur_voisins_2D_1point(epaisseurs3,diff,window_med)
    print('filtre différence voisins calculé')
    surf3D(v_temps,v_espace,epaisseurs4,'thickness map filtrée voisins, diff = ' + str(diff) + ', window = ' + str(window_med))
    thickness_map_pcolor(v_temps,v_espace,epaisseurs4,   'thickness map filtrée voisins, diff = ' + str(diff) + ', window = ' + str(window_med))
    
    #surf 3D nettoyée lissée
    epaisseurs5 = median_filter(epaisseurs4,(window,window))
    surf3D(v_temps,v_espace,epaisseurs5,'thickness map filtrée max voisins lissée, window = ' + str(window))
    thickness_map_pcolor(v_temps,v_espace,epaisseurs5,   'thickness map filtrée max voisins lissée, window = ' + str(window))
    
    #epaisseur minimale
    # print('epaisseur minimale =',np.min(epaisseurs),'nm')
    
    return(epaisseurs,epaisseurs1,epaisseurs2,epaisseurs3,epaisseurs4,epaisseurs5)
    

def get_thickness_in_point(file_bil,i_espace,i_temps,prop_max_outliers,prominence,distance,n):
    
    '''
    retourne salaire épaisseur calculée en un point d'espace et de temps
    1 lissage éventuel pour chaque spectre
    '''
    
    #récupère le datacube et ses infos
    [datacube,n_temps,n_espace,n_lambdas,v_temps,v_espace,v_lambdas] = get_datacube(file_bil)
        
    epaisseur,prop_outliers = calcul(datacube,v_lambdas,prominence,distance,i_temps,i_espace,prop_max_outliers,1,n)
    
    print('epaisseur =', epaisseur)
    print('proportion de outliers =', prop_outliers)
    
    return(epaisseur,prop_outliers)

def get_thickness_in_time(file_bil,i_espace,prop_max_outliers,prominence,distance,n):
    '''
    retourne vecteur des épaisseurs (temps) calculées en un point spatial pour plusieurs temps
    plot épaisseur en fonction du temps
    1 lissage éventuel pour chaque spectre
    '''
    
    #récupère le datacube et ses infos
    [datacube,n_temps,n_espace,n_lambdas,v_temps,v_espace,v_lambdas] = get_datacube(file_bil)

    epaisseurs1D = np.zeros(n_temps)
    datacube_jetés = []

    #boucle sur le temps    
    for i_temps in range(n_temps):
        print(i_temps)
        
        epaisseur,a = calcul(datacube,v_lambdas,prominence,distance,i_temps,i_espace,prop_max_outliers,2,n)
        epaisseurs1D[i_temps] = epaisseur
        
        if a:
            datacube_jetés.append(datacube[i_temps,i_espace])
        
    #plot épaisseur en fonction du temps
    plt.figure()
    plt.plot(v_temps,epaisseurs1D,'.')
    plt.xlabel('t (s)')
    plt.ylabel('h (nm)')
    
    input("appuyer sur n'importe quelle touche pour voir les spectres jetés")
    
    #calcul de la proportion de spectres gardés
    prop_spectres_gardes = len(epaisseurs1D[np.logical_not(np.isnan(epaisseurs1D))])/len(datacube[:,i_espace,:])
    print('proportion de spectres gardés =',prop_spectres_gardes)
    
    for i in range(len(datacube_jetés)):
        thickness_from_minmax(v_lambdas, datacube_jetés[i],n,prominence,distance,'ransac',True) 
    
    return(epaisseurs1D, prop_spectres_gardes)

def get_thickness_in_line(file_bil,i_temps,prop_max_outliers,prominence,distance,n):
    '''
    retourne vecteur des épaisseurs calculées en un instant, en espace
    plot épaisseur en fonction de l'espace
    1 lissage éventuel pour chaque spectre
    '''
    
    #récupère le datacube et ses infos
    [datacube,n_temps,n_espace,n_lambdas,v_temps,v_espace,v_lambdas] = get_datacube(file_bil)

    epaisseurs1D = np.zeros(n_espace)
    datacube_jetés = []

    #boucle sur le temps    
    for i_espace in range(n_espace):
        print(i_espace)
        
        epaisseur,a = calcul(datacube,v_lambdas,prominence,distance,i_temps,i_espace,prop_max_outliers,2,n)
        epaisseurs1D[i_espace] = epaisseur
        
        if a:
            datacube_jetés.append(datacube[i_temps,i_espace])
            
        
    #plot épaisseur en fonction du temps
    plt.figure()
    plt.plot(v_espace,epaisseurs1D,'.')
    plt.xlabel('x (pixel)')
    plt.ylabel('h (nm)')
    plt.show()
    
    input("appuyer sur n'importe quelle touche pour voir les spectres jetés")
    
    #calcul de la proportion de spectres gardés
    prop_spectres_gardes = len(epaisseurs1D[np.logical_not(np.isnan(epaisseurs1D))])/len(datacube[i_temps,:,:])
    print('proportion de spectres gardés =',prop_spectres_gardes)
    
    for i in range(len(datacube_jetés)):
        thickness_from_minmax(v_lambdas, datacube_jetés[i],n,prominence,distance,'ransac',True) 
    
    return(epaisseurs1D, prop_spectres_gardes)

def get_thickness_in_time_space(file_bil,indices_temps,indices_espace,file_out,prop_max_outliers,prominence,distance,n):
    '''
    - retourne matrice des epaisseurs calculees en temps et en espace
    - écrit dans un fichier csv le datacube des épaisseurs, le vecteur de temps et le vecteur d'espace
    '''
    
    [datacube,n_temps,n_espace,n_lambdas,v_temps,v_espace,v_lambdas] = get_datacube(file_bil)
    
    if indices_temps != np.array([]):
        v_temps = v_temps[indices_temps]
        datacube = datacube[indices_temps,:,:]
    if indices_espace != np.array([]):
        v_espace = v_espace[indices_espace]
        datacube = datacube[:,indices_espace,:]
        
    # datacube = datacube[::pas_temporel,::pas_spatial,:]
    
    epaisseurs2D = np.zeros([len(v_temps),len(v_espace)])
    
    for i_espace in tqdm(range(len(v_espace))):
        
        # print('i_espace =',pas*i_espace)
        
        for i_temps in range(len(v_temps)):
            
            epaisseur,a = calcul(datacube,v_lambdas,prominence,distance,i_temps,i_espace,prop_max_outliers,3,n)
            epaisseurs2D[i_temps,i_espace] = epaisseur
        
    #calcul de la proportion de spectres gardés
    prop_spectres_gardes = len(epaisseurs2D[np.logical_not(np.isnan(epaisseurs2D))])/np.size(epaisseurs2D)
    print('proportion de spectres gardés =',prop_spectres_gardes)
    
    #écriture des épaisseurs calculées dans un csv
    save_csv(epaisseurs2D,v_temps,v_espace,file_out)
    
    return(epaisseurs2D, v_temps, v_espace,prop_spectres_gardes)

def calcul(datacube,v_lambdas,prominence,distance,i_temps,i_espace,prop_max_outliers,num,n):
    '''
    prominence, distance : paramètres de la détections des extrema
    num : paramètre d'affichage pour debug pendant le calcul
    n : indice de réfraction
    '''

    #données en un point de l'espace et du temps
    datacube_i = datacube[i_temps,i_espace,:]

    #calcul de l'épaisseur
    try:
        if num == 1: #
            results = thickness_from_minmax(v_lambdas, datacube_i,n,prominence,distance,method='ransac',plot=True) #routine de oospectro
        else:
            results = thickness_from_minmax(v_lambdas, datacube_i,n,prominence,distance,'ransac',plot=False)
        
        epaisseur = results.thickness #en nm
        
        #si trop de outliers, on ne prend pas en compte l'épaisseur
        num_outliers = results.num_outliers
        num_inliers = results.num_inliers
        sum_liers = num_outliers + num_inliers
        prop_outliers = num_outliers/sum_liers
        
        if prop_outliers < prop_max_outliers:    
            if num == 2 :
                results = thickness_from_minmax(v_lambdas, datacube_i,n,prominence,distance,'ransac',plot=False)
                
            return(epaisseur,False)
            
        else:
            if not(num == 3):
                print("pas d'épaisseur")
                epaisseur = None
            if num != 1:
                return(epaisseur,True)
            
        if num == 1:
            return(epaisseur,prop_outliers)
    
    except (ValueError):
        print("ValueError1")
        if num == 1:
            results = thickness_from_minmax(v_lambdas, datacube_i,n,prominence,distance,'linreg',plot=True) #routine de oospectro
        else:
            results = thickness_from_minmax(v_lambdas, datacube_i,n,prominence,distance,'linreg',plot=False)
        
        return(results.thickness,False)
        
    except(AttributeError):
        epaisseur = None
        if not(num == 3):
            print("ValueError")
        return(epaisseur,False)
            
def filtre_median2D_sur_nan(matrice):
    '''Pour chaque valeur NaN dans la matrice, prend la médiane des points voisins
    afin de ne plus avoir de NaN dans la matrice'''
    
    [n,m] = np.shape(matrice)
    matrice_out = copy.deepcopy(matrice)
    
    for i in range(n):
        for j in range(m):
            window = 1
            while np.isnan(matrice_out[i,j]):
                matrice_autour = matrice_out[i-window if i-window>0 else 0:i+window+1,j-window if j-window>0 else 0:j+window+1]
                matrice_out[i,j] = np.median(matrice_autour[np.logical_not(np.isnan(matrice_autour))])
                window += 1
                            
    return(matrice_out)

def filtre_epaisseur_max_2D_1point(matrice,epaisseur_max):
    '''Pour chaque point de la matrice, teste si l'épaisseur en ce point dépasse
    'epaisseur_max'. Si oui, remplace l'épaisseur de ce point par la médiane
    des épaisseurs des points voisins à une distance qui augmente jusqu'à ce que
    l'épaisseur soit plus petite que 'epaisseur_max' '''
    
    [n,m] = np.shape(matrice)
    matrice_out = copy.deepcopy(matrice)
    
    for i in range(n):
        for j in range(m):
            window = 1
            while matrice_out[i,j] > epaisseur_max:
                matrice_autour = matrice_out[i-window if i-window>0 else 0:i+1,j-window if j-window>0 else 0:j+window+1]
                matrice_out[i,j] = np.median(matrice_autour)
                window += 1
    
    return(matrice_out)

def filtre_epaisseur_voisins_2D_1point(matrice,diff,window):
    '''Pour chaque point de la matrice, teste si l'épaisseur en ce point dépasse
    de plus de 'diff' la médiane des points premiers voisins dans la matrice.
    Si oui, le point prend la valeur de cette médiane.
    Retourne la matrice modifiée.'''

    [n,m] = np.shape(matrice)
    matrice_out = copy.deepcopy(matrice)
    
    for i in range(n):
        for j in range(m):
            
            autour = vecteur_autour(matrice_out,i,j,window)
            
            if matrice_out[i,j] > np.median(autour) + diff or matrice_out[i,j] < np.median(autour) - diff:
                matrice_out[i,j] = np.median(autour) 
    
    return(matrice_out)

def plusieurs_lignes(epaisseurs,v_temps,v_espace,temps,offset_colormap,lignes,e0):
    '''affiche l'épaisseur en fonction de l'espace pour les instants dont les
    indices sont contenus dans 'temps' et la même chose redimensionnée avec les
    scalings d'Aradian'''
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    
    #colormap
    cmap = plt.cm.get_cmap("Purples")
    if temps[2] - temps[1] == temps[1] - temps[0] or temps[2] - temps[1] == temps[1] - temps[0] + 1 or temps[2] - temps[1] == temps[1] - temps[0] - 1 : #instants ditribués en lin
        norma = mpl.colors.Normalize(vmin=np.min(temps)-offset_colormap, vmax=np.max(temps))
    else: #instants ditribués en log
        norma = mpl.colors.Normalize(vmin=np.min(np.log(temps))-offset_colormap, vmax=np.max(np.log(temps)))
    sm = mpl.cm.ScalarMappable(norm=norma, cmap=cmap)
    sm.set_array([])
    
    
    for i in range(len(temps)):
        if temps[2] - temps[1] == temps[1] - temps[0] or temps[2] - temps[1] == temps[1] - temps[0] + 1 or temps[2] - temps[1] == temps[1] - temps[0] - 1 : #instants ditribués en lin
            
            # epaisseurs[temps[i],:] = savgol_filter(epaisseurs[temps[i],:],51,3)
            plt.plot(v_espace,epaisseurs[temps[i],:]/1000,'.-' if lignes else '.', color=cmap(norma(temps[i])),label='t = ' + format(v_temps[temps[i]]/3600, ".1f") + ' h') #str(v_temps[temps[i]])

        else: #instants ditribués en log
        
           plt.plot(v_espace,epaisseurs[temps[i],:]/1000,'.-' if lignes else '.', color=cmap(norma(np.log(temps[i]))),label='t = ' + format(v_temps[temps[i]]/3600, ".1f") + ' h') #str(v_temps[temps[i]])
         
    plt.xlabel('x (mm)')
    plt.ylabel('h ($\mu$m)')
    # plt.title('épaisseur sur une ligne pour plusieurs instants')
    plt.legend()
    
    #rescaling
    g = 9.81
    
    ############# à modifier
    # rho = 10**3 #eau
    rho = 0.975*10**3 #silicone
    
    # gamma = 72*10**(-3) #eau
    # gamma = 25*10**(-3) #eau savonneuse
    gamma = 20*10**(-3) #huile silicone
    
    # eta = 10**(-3) #eau
    eta = 30 * 10**(-3) #huile silicone
    
    lc = np.sqrt(gamma/(rho*g)) #longueur capillaire
    
    R = 0.74 * 10**(-3)/2 #m
    
    r = lc #extension du bord de plateau
    c = R/lc**2 #courbure
    #############
    
    eps = e0/r
    A = 1.591
    v_star = gamma/eta
    t_relax = 3*r**4/(v_star*e0**3)
    T = v_temps/t_relax
    # w = eps*A/(np.sqrt(2)*r*c)*T**(-1/4)
    # h = eps*A**2/(2*r*c)*T**(-1/2)
    
    
    E = epaisseurs/e0
    F = E
    X = v_espace/r
    
    #scaling zone centrale
    # plt.figure()
    
    # for i in range(len(temps)):
        
    #     U = X/(np.sqrt(2)*T[temps[i]]**(1/4))
        
    #     plt.plot(U,F[temps[i]],'.-' if lignes else '.',color=cmap(norma(np.log(temps[i]))),label='T = ' + format(T[temps[i]], ".1e"))
    #     plt.xlabel('U')
    #     plt.ylabel('F')
    #     plt.legend()
    #     #plt.legend([T[temps[i]] for i in range(len(temps))])
        
    # #scaling pinch
    # plt.figure()
    
    # for i in range(len(temps)):
        
    #     S = E/h[temps[i]]
    #     xi = X/w[temps[i]]
        
    #     plt.plot(xi,S[temps[i]],'.-' if lignes else '.',color=cmap(norma(np.log(temps[i]))),label='T = ' + format(T[temps[i]], ".1e"))
    #     plt.xlabel(r'$\xi$')
    #     plt.ylabel('S')
    #     plt.legend()
    #     #plt.legend([T[temps[i]] for i in range(len(temps))])
    
    
def vecteur_autour(matrice,i,j,window):
    ''''retourne la liste des points de la matrice autour du point d'intérêt
    à une certaine distance 'window', et qui ont déjà été traités'''
    
    autour=[]
    
    if (i,j)==(0,0):
        # autour = matrice[i:i+window+1,j:j+window+1]
        # autour = autour.flatten()
        # # autour = list(autour)
        autour = list(matrice[i:i+window+1,j:j+window+1].flatten())
        autour = autour[1:]
    else:
        for a in range(i-window if i-window>0 else 0,i+1):
                    for b in range(j-window if j-window>0 else 0,j+window+1 if j+window+1<=np.shape(matrice)[1] else np.shape(matrice)[1]): 
                        if a < i:
                            autour.append(matrice[a,b])
                        else:
                            if b < j:
                                autour.append(matrice[a,b])
    return(autour)

def save_csv(epaisseurs,v_temps,v_espace,file_csv):
    
    with open(file_csv, "w", newline='') as data:
        writer = csv.writer(data,delimiter=' ')
        for i in range(np.shape(epaisseurs)[0]):
            writer.writerow(epaisseurs[i])
        writer.writerow(v_temps)
        writer.writerow(v_espace)

def h_vs_t(files,i_espace):
    #Marina
    
    n = len(files)
    epaisseurs = []
    v_temps = []
    
    plt.figure()
    for i in range(n):
        e, v_t, _, _ = infos_from_csv(files[i])
        epaisseurs.append(e[:,i_espace])
        v_temps.append(v_t)
        plt.plot(v_temps[i],epaisseurs[i],'.')
    
    plt.xlabel('t (s)')
    plt.ylabel('h (nm)')
    
def h_vs_H(files):
    #Marina
    '''3 vitesses, 1 position'''
    
    position = int(files[0][files[0].find('Pos' )+9 : files[0].find('cm')])
    
    files_names = [files[i][files[i].find('TEST'):files[i].find('TEST')+5] + files[i][files[i].find('cm-')+3:-4] for i in range(len(files))]
    
    vitesses = [20,100,160]
    
    # h_max = 190
    
    n = len(files)
    
    #epaisseurs, v_temps,v_espace,H contiennent les infos pour tous les files
    epaisseurs = []
    v_temps = []
    v_espace = []
    H = []
    # i_temps_max = []
    
    #récupère les informations des files
    for i in range(n):
        _, v_t, v_e, _ = infos_from_csv(files[i])
        
        v_espace.append(v_e)
        v_temps.append(v_t) #temps depuis le début de la génération
        h = vitesses[i] * v_temps[i] #hauteur de film généré
        # i_temps_max.append(np.where(h > h_max)[0][0])
        # H.append(h[:i_temps_max[i]])
        H.append(h)
    
    #i_espace est le plus petit indice d'espace en commun avec tous les files
    v_espace = np.array(v_espace,dtype=object)
    m = np.max([np.min(v_espace[i]) for i in range(n)])
    i_espace = [np.where(v_espace[i] == m)[0] for i in range(n)]
    
    #colormap
    cmap = plt.cm.get_cmap("Blues")
    norma = mpl.colors.Normalize(vmin=0-1, vmax=2)
    sm = mpl.cm.ScalarMappable(norm=norma, cmap=cmap)
    sm.set_array([])
    
    #récupérer le reste des informations des files + tracer
    plt.figure()
    for i in range(n):
        e, _, _, _ = infos_from_csv(files[i])
        # epaisseurs.append(e[:i_temps_max[i],i_espace[i]]) #en temps pour un point d'espace
        epaisseurs.append(e[:,i_espace[i]]) #en temps pour un point d'espace
        plt.plot(H[i],epaisseurs[i],'.',label= 'V = ' + str(vitesses[i]) + "cm/s : " + str(files_names[i]),color=cmap(norma(i)))

    plt.xlabel('H (cm)')
    plt.ylabel('h (nm)')
    plt.title(str(position) + 'cm au-dessus du bain, ' + '{:.2f}'.format(m) + 'cm du milieu du film')
    plt.legend()
    
    return(v_espace)

def create_param_file(file_csv,prominence):
    
    _,v_temps,v_espace,_ = infos_from_csv(file_csv)
    
    prominence = np.ones(len(v_espace)) * prominence
    window = np.zeros(len(v_espace))
            
    with open(file_csv[:-4] + ',prominence.csv', "w", newline='') as data:
            writer = csv.writer(data,delimiter=' ')
            for i_temps in range(len(v_temps)):
                writer.writerow(prominence)
                
    with open(file_csv[:-4] + ',window.csv', "w", newline='') as data:
            writer = csv.writer(data,delimiter=' ')
            for i_temps in range(len(v_temps)):
                writer.writerow(window)

def save_param_file(file_name,prominence,window):
    
    n_temps = len(prominence[:,0])
            
    with open(file_name[:-4] + ',prominence.csv', "w", newline='') as data:
            writer = csv.writer(data,delimiter=' ')
            for i_temps in range(n_temps):
                writer.writerow(prominence[i_temps,:])
                
    with open(file_name[:-4] + ',window.csv', "w", newline='') as data:
            writer = csv.writer(data,delimiter=' ')
            for i_temps in range(n_temps):
                writer.writerow(window[i_temps,:])
                

def read_param_file(file_prominence,file_window,file_csv):
    
    _,v_temps,v_espace,_ = infos_from_csv(file_csv)

    param = np.zeros([len(v_temps),len(v_espace),2])
    with open(file_prominence, newline='') as fichier:
        data = csv.reader(fichier,quoting = csv.QUOTE_NONNUMERIC,delimiter=' ',)
        i_temps = 0
        for row in data:
            param[i_temps,:,0] = row
            i_temps += 1
            
    with open(file_window, newline='') as fichier:
        data = csv.reader(fichier,quoting = csv.QUOTE_NONNUMERIC,delimiter=' ',)
        i_temps = 0
        for row in data:
            param[i_temps,:,1] = row
            i_temps += 1
            
    return(param)

def analyse_std(epaisseurs,largeur):
    
    n_temps  = len(epaisseurs[:,0])
    n_espace = len(epaisseurs[0,:])
    
    std = np.zeros([n_temps,n_espace])

    for i_temps in range(n_temps):
        for i_espace in range(n_espace):
            std[i_temps,i_espace] = np.std(epaisseurs[i_temps,i_espace - largeur:i_espace + largeur])

    mask_std = std > 400
    nbr_recalcul = np.count_nonzero(mask_std)

    return(std,mask_std,nbr_recalcul)