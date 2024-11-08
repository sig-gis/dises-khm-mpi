#Este archivo contiene funciones que facilitan la construcción de mapas
import pandas as pd
import geopandas as gpd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import sys
import numpy as np
import mapclassify
import contextily as ctx
from shapely.geometry import box
import mapclassify
from shapely.ops import cascaded_union
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


#%%
def diccionario_rutas(path_OneDrive='/Users/Daniel/Library/CloudStorage/OneDrive-SharedLibraries-VestigiumMétodosMixtosAplicadosSAS/'):

    """Retornan un diccionario con las rutas a los archivos
    ---
    path_OneDrive: Ruta al folder de OneDrive que contiene la carpeta "programación".
    """
    archivo_rutas = 'MMC - General - Programacion/funciones/rutas.xlsx'
    
    df = pd.read_excel(path_OneDrive+archivo_rutas)
    df['Path'] = df['Path'].apply(lambda x: path_OneDrive + x)
    paths = dict(zip(df['Key'], df['Path']))
            
    #Paths OS agnostic paths
    for i in paths:
        paths[i] = os.path.abspath(paths[i])

    return paths

#%%Se crea una función para cargar una sola capa a la vez, en caso que se necesite
def cargar_capa_individual(filename, **kwargs):
    """Returns a gdf with the shape loaded and adjusted to the CRS:3116 (Bogota-centred metric system)
    Parameters:
        filename : str, path object or file-like object
            Either the absolute or relative path to the file or URL to
            be opened, or any object with a read() method (such as an open file
            or StringIO)
        mask : dict | GeoDataFrame or GeoSeries | shapely Geometry, default None 
            Filter for features that intersect with the given dict-like geojson
            geometry, GeoSeries, GeoDataFrame or shapely geometry.
            CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
            Cannot be used with bbox.
        **kwargs :
            Keyword args to be passed to the `read_file` method
            in the geopandas library when opening the file. The mask and bbox arguments might be especially useful"""
      
    if filename[-3:] != 'shp':  # Se excluyen otros tipos de archivos
        return print('La ruta no apunta a un archivo Shape')
    capa = gpd.read_file(filename, encoding='utf-8', **kwargs)
   # print('Ajustando capa {}'.format(filename.split('/')[-1:][0]))
    # Se unifica todo en mismo crs
    capa = capa.to_crs({'init': 'EPSG:4326'})
    # Se corta para que los shapes no excedan las dimensiones de cobertura del crs métrico centrado en Bogotá
    capa = capa.cx[-79.1:-66.87, -4.23:13.68]
    # Se pasa todo a un crs métrico centrado en Bogotá, para aumentar la precisión de los cálculos y para que queden en metros
    capa = capa.to_crs({'init': 'EPSG:3116'})   
    #If a mask is used, clip the final layer to avoid possible inaccuracies in the read_file method of geopandas
    if 'mask' in kwargs.keys():
        mask = kwargs.get('mask').to_crs({'init': 'EPSG:3116'})
        mask['geometry'] = mask.buffer(0.01)
        if filename.split('/')[-1:][0] in ['VRDS_MPIO_POLITICO.shp', 
                                           'CRVeredas_2017.shp',
                                           'base_veredas.shp', 'Resguardos_Indigenas.shp', 
                                           'Consejos_comunitarios.shp', 
                                           'Lotes_18022019.shp']:
            capa = capa[capa.area>0] #Drop bordering polygons that have no area after clipping
    #        print('Empty polygons dropped')
            capa['geometry'] = capa.buffer(0.01) #Ajust geometries to avoid Rtree errors
            #print('Buffer created to avoid errors while joining')
            capa = gpd.clip(capa, mask)
     #       print('Final layer clipped to the mask extent for accuracy')
            return capa    
        capa = gpd.clip(capa, mask)
      #  print('Final layer clipped to the mask extent for accuracy')
        return capa
    #print('Layer adjusted')
    return capa

#Utility functions to facilitate the creation of more complex ones

def indice_municipio(cod_mpio, path_OneDrive):
    """Retorna la fila que corresponde al municipio en el archivo 'municipos' del diccionario paths.
        Parametros:
            cod_mpio (str): Código del municipio en formato string"""
    paths = diccionario_rutas(path_OneDrive=path_OneDrive)
    inds = pd.read_csv(paths['municipios_indices'], 
                      dtype={'DPTOMPIO':str},
                      index_col=0)
    ind = inds[inds['DPTOMPIO']==cod_mpio]

    return ind.index.to_list()[0]+1

def get_bounding_box(gdf):
    """Returns a gdf with only one row that is box that encloses all geometries in the initial gdf
    ------
    Parameters:
        gdf (gdf): Geopandas.GeoDataFrame object"""
    bbx = box(*gdf.total_bounds)
    bbx = gpd.GeoDataFrame(geometry=[bbx])
    bbx.crs = gdf.crs
    return bbx

#This is a function to create custom, automatic maps for each municipality in Colombia


def mapa_general(shape,
                path_OneDrive,
                año='2019',
                show_coca=True,
                coca='coca19', 
                emf_ano='2019', 
                lgnd_loc=False,
                labels=False,
                labels_col='.',
                vias=True, 
                figsize=(12, 12), 
                vrds=True, 
                rios=True,
                aurbana=True,
                pnn=True,
                minas=True,
                pnis=True,
                cca=True,
                rsg=True,
                zf=True,
                zrf=True,
                palma=False,
                evoa=False,
                title='.',
                show=True,
                filename='.', 
                dpi=100, 
                 n_col=1,
                 marco=True,
                **kwargs):
    """Retorna un mapa de matplot lib con el municipio y las diferentes capas resaltadas.
            Parametros:
                path_OneDrive (str): replace with your own path to make the function work on any computer. 
                shape (gdf): Geodataframe with the zone of interest.
                show_coca (Bool): True to show the layer of illegal coca fields. Default is True.  
                coca (str): Nombre las capas de coca de 2001-19, escrito como cocaYY
                emf_ano (str): Year for the EMF (manual erradication) layer (data from 2016)
                labels (Bool): add labels for polygons in shape. Has to be used with labels_col. Default is false. 
                labels_col (str): Name of the column in the shape df that has the labels to be added.
                vrds (Bool): True to show the borders and names of veredas. Default is true. 
                minas (Bool): True to show the points with accidents due to Land Mines (last 5 years). Default is true. 
                evoa (Bool): True to show illegal gold mining areas (2019). Default is False.
                filename: str or path-like or file-like A path, or a Python file-like object, or possibly some backend-dependent object such as `matplotlib.backends.backend_pdf.PdfPages`
                dpi: float or 'figure', default: :rc:`savefig.dpi`
                The resolution in dots per inch.  If 'figure', use the figure's
                dpi value.
                figsize : tuple of integers (default None)
                        Size of the resulting matplotlib.figure.Figure. If the argument
                        axes is given explicitly, figsize is ignored."""
    paths = diccionario_rutas(path_OneDrive=path_OneDrive)
    df=shape
    bbx = get_bounding_box(shape)
    
    #df = mpios[mpios['DPTOMPIO']==cod_mpio]
    fig, ax = plt.subplots(figsize=figsize)

    #Get the bounding box to cut the other layers
    #df = cargar_capa_individual(paths['municipios'], maks=bbx)
    df.plot(ax=ax, color='white', alpha=0.01)
   
    #Load and add to the map jurisdiction layers
    legend_patches = [] #List to store legend patches
    
    if pnn==True:
        pnn = cargar_capa_individual(paths['pnn'], mask=bbx)
        pnn.plot(ax=ax, color='limegreen', alpha=0.5)
        #Add label        
        pnn.boundary.plot(ax=ax, color='forestgreen')
        pnn_legend = mpatches.Patch(color='limegreen', label='Parque Nacional') 
        legend_patches = legend_patches + [pnn_legend]
        #Add label
        pnn.apply(lambda x: ax.annotate(text=x['NOM_PARQ'], 
                                         xy=x.geometry.centroid.coords[0], 
                                         ha='center', 
                                         color='forestgreen', 
                                         fontsize=7, 
                                        alpha=0.7),
                                        axis=1)
        
    if cca==True:
        cca = cargar_capa_individual(paths['consejos'], mask=bbx)
        cca.plot(ax=ax, color='peru', alpha=0.5)
        #Add label
        cca.apply(lambda x: ax.annotate(text=x['NOMBRE'], 
                                         xy=x.geometry.centroid.coords[0], 
                                         ha='center', 
                                         color='saddlebrown', 
                                         fontsize=7, 
                                        alpha=0.7),
                   axis=1)
        cca.boundary.plot(ax=ax, color='saddlebrown')
        cca_legend = mpatches.Patch(color='peru', label='CC Afro') 
        legend_patches = legend_patches + [cca_legend]
    if rsg==True:
        rsg = cargar_capa_individual(paths['resguardos'], mask=bbx)
        rsg.plot(ax=ax, color='palevioletred', alpha=0.8)
        #Add label        
        rsg.boundary.plot(ax=ax, color='purple')
        rsg_legend = mpatches.Patch(color='palevioletred', label='Resguardo') 
        legend_patches = legend_patches + [rsg_legend]
        
    if zrf==True:
        zrf = cargar_capa_individual(paths['zonificacion_reservas'], mask=bbx)
        zrf.plot(ax=ax, color='darkgreen', alpha=0.5)
        #Add label
        zrf.boundary.plot(ax=ax, color='darkgreen')
        zfr_legend = mpatches.Patch(color='darkgreen', label='Reserva Forestal') 
        legend_patches = legend_patches + [zfr_legend]
        
    if zf==True:
        zf = cargar_capa_individual(paths['zonas_futuro'], mask=bbx)
        zf.plot(ax=ax, color='lightsteelblue', alpha=0.5)
        if zf.area.sum() > 0: #Only draw the patch if the layer exist in the map
            zf_legend = mpatches.Patch(color='lightsteelblue', label='Zona Futuro') 
            legend_patches = legend_patches + [zf_legend]
    #Se establecen los límites del mapa con base en los límites del municipio
    b = df.total_bounds
    ax.set_xlim(b[0]-100, b[2]+100)
    ax.set_ylim(b[1]-100, b[3]+100)

    #Se cargan las capas
    if show_coca==True:
        coca_shape = cargar_capa_individual(paths[coca], mask=bbx) #Coca
        coca_shape.plot(ax=ax, column='areacoca', cmap='Reds', alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=mpl.colors.Normalize(0,100))
        fig.colorbar(sm, ax=ax, label='Coca Hectares', aspect=60, fraction=0.046, pad=0.04)
        
    #Se agrega la capa de evoa (Minería ilegal de oro de aluvión)
    if evoa==True:
        color_map='YlOrBr' #Set the color map for the EVOA layer
        if show_coca==True: #Change the color map if coca is to be shown to avoid confusion with the colors 
            color_map='spring' 
        evoa = cargar_capa_individual(paths['evoa'], mask=bbx) #Coca
        evoa.plot(ax=ax, column='E_ILICITA', cmap=color_map, alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=mpl.colors.Normalize(0,100))
        fig.colorbar(sm, ax=ax, label='Hectáreas de EVOA', aspect=60, fraction=0.046, pad=0.04)
    
    if vrds==True:
        vrds = cargar_capa_individual(paths['vrds'], mask=df) #Veredas
        vrds.plot(ax=ax, edgecolor='grey', alpha=0.001)  
        #Labels for veredas
        vrds.apply(lambda x: ax.annotate(text=x['NOMBRE_VER'], xy=x.geometry.centroid.coords[0], ha='center', color='grey', fontsize=7),axis=1);
        vrds.boundary.plot(ax=ax, color='grey', linestyle='--') #Draw boundaries for veredas
    #EMF
    try:
        emf = cargar_capa_individual(paths['emf'], mask=bbx)
        emf['FECHA'] = pd.to_datetime(emf['FECHA'])
        emf.set_index('FECHA', inplace=True)
        emf = emf[emf_ano]
        emf.plot(ax=ax, color='orange', marker=".",markersize=10)
        emf_legend = mpatches.Patch(color='orange', label='Erradicación F.')
        legend_patches = legend_patches + [emf_legend]
    except:
        print('Area sin emf')
    #Add the rivers layer
    if rios==True:
        rios = cargar_capa_individual(paths['rios'], mask=bbx)
        rios.plot(ax=ax, color='turquoise')    
    if vias==True:
        vias = cargar_capa_individual(paths['vias'], mask=bbx)
        vias.plot(ax=ax, color='yellow')  
        #if vias.area.sum() > 0: #Only draw the patch if the layer exist in the map
        vias_legend = mpatches.Patch(color='yellow', label='Roads')
        legend_patches = legend_patches + [vias_legend]
    if aurbana==True:
        aurbana = cargar_capa_individual(paths['aurbana'], mask=bbx)
        aurbana.plot(ax=ax, color='grey')
        if aurbana.area.sum() > 0: #Only draw the patch if the layer exist in the map
            aurbana_legend = mpatches.Patch(color='grey', label='Urban area') 
            legend_patches = legend_patches + [aurbana_legend]
    if minas == True:
        minas = cargar_capa_individual(paths['map'], mask=bbx)
        minas.plot(ax=ax, color='red', marker="*",markersize=30, alpha=1)
        if minas.shape[0] > 0: #Only draw the patch if the layer exist in the map
            red_star = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=10, label='Minas Ant.') #Add labels
            legend_patches = legend_patches + [red_star]
    if (pnis == True) & (16<int(coca[-2:])<22):
        pnis = cargar_capa_individual(paths['lotes2020'], mask=bbx)
        pnis.plot(ax=ax, color='#987db7', marker=".",markersize=10)
        if pnis.shape[0] > 0: #Only draw the patch if the layer exist in the map
            pnis_legend = mpatches.Patch(color='#987db7', label='PNIS beneficiaries')
            legend_patches = legend_patches + [pnis_legend] 
    if palma == True:
        palma = cargar_capa_individual(paths['palma'], mask=bbx)
        palma.plot(ax=ax, color='darkgreen', marker=".",markersize=10, alpha=0.8)
        palma_legend = mpatches.Patch(color='darkgreen', label='Palm crops (2019)')
        legend_patches = legend_patches + [palma_legend]
    #df2.plot(ax=fig, figsize=(10, 10), color='grey', alpha=0.5)

    #Titulo

    if title!= 'Mapa de diagnóstico zonal':
        ax.set_title(title, fontdict={'fontsize': 16})

    #Se grafica el lote con base en las coordenadas recibidas
    try:
        ctx.add_basemap(ax=ax, source=ctx.providers.Stamen.Terrain, crs=df.crs.to_string(), alpha=0.5)
        ax.set_axis_off()
    
    except:   
        try:
            ctx.add_basemap(ax=ax, source=ctx.providers.Stamen.TerrainBackground, crs=df.crs.to_string(), alpha=0.5)
            ax.set_axis_off()
        except:
            ctx.add_basemap(ax=ax, source=ctx.providers.CartoDB.Voyager, crs=df.crs.to_string())
            ax.set_axis_off()
    
    df.boundary.plot(ax=ax, color='black')  #Bounds
    
    #Add labels for the primary shape
    if labels==True:
        try:
            df.apply(lambda x: ax.annotate(text=x[labels_col], 
                                   xy=x.geometry.centroid.coords[0], 
                                   ha='center', color='black', 
                                   fontsize=10),axis=1)
        except:
            print('Labels for primary layer could not be added. Check parameters.')

    
    #Legenda
    #veredas_municipo = mpatches.Patch(color='green', label='Coca en 20'.format(coca[-2:]))
    #vereda = mpatches.Patch(color='grey', label='Vereda en la que se ubica el lote')
   
    plt.legend(handles=legend_patches, loc=lgnd_loc, ncol=n_col, frameon=marco)
    if filename != '.':    
        plt.savefig(filename, dpi=dpi)
    
    return ax   

def mapa_municipal(cod_mpio, 
                    path_OneDrive, 
                   año=2019, 
                   filename='.', 
                   dpi=100, 
                   **kwargs):
    """Retorna un mapa de diagnóstico del municipio seleccionado
    Parameters:
    -----------
        cod_mpio (str): Código del municipio en formato string.
        filename: str or path-like or file-like
            A path, or a Python file-like object, or
            possibly some backend-dependent object such as
            `matplotlib.backends.backend_pdf.PdfPages`
        dpi: float or 'figure', default: :rc:`savefig.dpi`
            The resolution in dots per inch.  If 'figure', use the figure's
            dpi value.
        año: Year to be displayed in the title of the chart. 
        kwargs: All other arguments from mapa_general()
        """
    paths = diccionario_rutas(path_OneDrive=path_OneDrive)
    row= indice_municipio(cod_mpio, path_OneDrive=path_OneDrive)
    df = cargar_capa_individual(paths['municipios'], rows=slice(row-1,row))
    municipio = df['NOMB_MPIO'].iloc[0].capitalize()
    depto = df['NOM_DEP'].iloc[0].capitalize()
    if 'title' not in kwargs:
        kwargs['title'] = 'Mapa diagnóstico del municipio de {}, {} ({})'.format(municipio, depto, año)
    ax = mapa_general(df, path_OneDrive=path_OneDrive, **kwargs)
    if 'title' in kwargs:
        ax.set_title(kwargs.get('title', 'Mapa diagnóstico'), fontdict={'fontsize': 16})
    df.apply(lambda x: ax.annotate(text=x['NOMB_MPIO'], xy=x.geometry.centroid.coords[0], ha='center', color='black', fontsize=10),axis=1)
    if filename != '.':    
        plt.savefig(filename, dpi=dpi)
    return ax

def ubicar_otros_beneficiarios_pnis(vrds, lts, df2):
    
    """Retorna el df de pagos del PNIS con algunos de los beneficiarios que no tienen lotes ubicados
        ---
        Parámetros:
            vrds (gdf): geopandas dataframe con la capa de veredas del DANE
            lts (gdf): geopandas dataframe con los lotes georreferenciados del PNIS
            df2: (df): Dataframe general de registro recibido de la DSCI con información de los beneficiarios del PNIS
    """
   

    lts = gpd.sjoin(lts, vrds[['CODIGO_VER', 'geometry']]) #Se le pega la información de las veredas a los lotes

    #Seleccionar unicamente el lote mas grande de cada beneficiario
    ids = lts.groupby(['CUB'])['A_ERRA'].idxmax()
    lts = lts.loc[ids]

    #Se crea una nueva variable con la vereda y el código DANE juntos para evitar errores
    def vereda_unida(x):
        return str(x['CODIGO DANE']) + '_' + x['VEREDA']
    
    df2['VEREDA_PNIS'] = df2.apply(vereda_unida, axis=1)

    #Asignar las veredas del DANE al df de general de registro
    df4 = df2[['CUB', 'VEREDA_PNIS']].merge(lts[['CUB', 'CODIGO_VER']], on='CUB', how='left')
    df5 = df4[['VEREDA_PNIS','CODIGO_VER']].dropna().drop_duplicates()
    df6 = df4[['VEREDA_PNIS','CODIGO_VER']].dropna()
    df7 = df6.groupby(['VEREDA_PNIS', 'CODIGO_VER']).size().reset_index() #df con la frequencia de veredas DANE por veredas PNIS
    df7.columns=['VEREDA_PNIS', 'CODIGO_VER', 'FRECUENCIA']
    df7 = df7.loc[df7.groupby('CODIGO_VER')['FRECUENCIA'].idxmax()] #Unicamente las observaciones mas frecuentes
    vrds_pnis_dane = dict(zip(df7['VEREDA_PNIS'], df7['CODIGO_VER'])) #Se crea un diccionario para asignar las veredas PNIS a veredas DANE cuando no hay cruce geografico

    #Nota: La DSCI también suministró un Shape de las veredas PNIS que el programa a definido. Sin embargo, la siguiente tabla muestra que hay veredas PNIS que tienen beneficiarios que están hasta en 16 veredas DANE diferentes. Esto indica que las veredas PNIS son mucho más grandes que las del DANE. Por lo tanto, utilizar las veredas PNIS produciría calculos más imprecisos que con las veredas del DANE, de manera que se trabaja con esta última.

    reg = df2.merge(lts[['CUB', 'CODIGO_VER']], on='CUB', how='left')
    reg['CODIGO_VER'].fillna(True, inplace=True)

    def asignar_veredas(x, dic):
        """Esta función asigna veredas para los casos en los que no hay cruce geográfico pero sí hay otro beneficiario asignado a la misma vereda PNIS que tenía un lote."""
        if x['CODIGO_VER']!=True:
            return x['CODIGO_VER']
        if x['VEREDA_PNIS'] in dic.keys():
            return dic[x['VEREDA_PNIS']]
        else:
            return np.nan

    reg['CODIGO_VER'] = reg.apply(lambda x: asignar_veredas(x, vrds_pnis_dane), axis=1)
    return reg
