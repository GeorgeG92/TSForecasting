import math
import pandas as pd
import os
from pylab import rcParams
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings


class clusterAnalysis():
    def __init__(self, df, args, explore=False, exportpath=os.path.join('..','output','Clustering')):
        print("Cluster Analysis")
        warnings.filterwarnings("ignore")
        self.exportpath = exportpath
        df = self.CleanData(df)
        if explore!=False:
            self.clusterExploration(df, 'source')
            self.clusterExploration(df, 'destination')
        df = self.clusterData(df)
        self.drawClustersOnMap(df, 'source')
        self.drawClustersOnMap(df, 'destination')

    def clusterExploration(self, df_geo, kind):
        performanceList = []
        K = range(5,30)
        """
        Trains K-means to the coordinates and exports elbow diagram to outputfolder
        Arguments:
            df: Input dataframe
            kind (string): 'source' or 'destination'
            outputfolder (string): folder to export elbow diagram
        Returns:
            data (pd.DataFrame): new dataframe with two new columns
            with the predicted source and destination based on the clustering
            algorithm
        """
        assert kind=='source' or kind=='destination'
        print("\tExploring K for KMeans for "+str(kind)+" coordinates...")
        df_geo = df_geo[[str(kind)+'_longitude', str(kind)+'_longitude']]

        # Explore K
        for k in K:
            kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(df_geo)
            performanceList.append(kmeanModel.inertia_)

        # Plot the elbow
        plt.plot(K, performanceList, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Error vs K in '+str(kind)+' coordinates')
        if not os.path.exists(self.exportpath):
            os.mkdir(outputfolder)
        plt.savefig(os.path.join(self.exportpath,'elbow_rule_'+str(kind)+'.png'))
        plt.close()

    def CleanData(self, df):
        """
        Cleans nan values in columns list to enable K-means
        Arguments:
            df: Input dataframe
            columns (list): 'source' and 'destination'
        Returns:
            The clean DataFrame
        """
        df = df.dropna(subset=['source_latitude', 'source_longitude', 'destination_longitude', 'destination_latitude'])
        return df

    def clusterData(self, df, k=15):                                    # default after elbow rule
        print("\tPerforming K-means for both source and destination coordinates with K="+str(k))
        kmeans_model = KMeans(n_clusters=k, n_jobs=-1).fit(df[['source_latitude', 'source_longitude']])
        df['sourceCluster'] = kmeans_model.predict(df[['source_latitude', 'source_longitude']])

        kmeans_model = KMeans(n_clusters=k, n_jobs=-1).fit(df[['destination_latitude', 'destination_longitude']])
        df['destinationCluster'] = kmeans_model.predict(df[['destination_latitude', 'destination_longitude']])
        return df

    def calculateBoarders(self, df, decimals=4):
        # Needed to download Lima map from https://www.openstreetmap.org/export#map=5/51.500/-0.100

        sourceDict = {'longitude_max': np.around(df['source_longitude'].max(), decimals=decimals),
                      'longitude_min': np.around(df['source_longitude'].min(), decimals=decimals),
                      'latitude_max': np.around(df['source_latitude'].max(), decimals=decimals),
                      'latitude_min': np.around(df['source_latitude'].min(), decimals=decimals)}

        destinationDict = {'longitude_max': np.around(df['destination_longitude'].max(), decimals=decimals),
                      'longitude_min': np.around(df['destination_longitude'].min(), decimals=decimals),
                      'latitude_max': np.around(df['destination_latitude'].max(), decimals=decimals),
                      'latitude_min': np.around(df['destination_latitude'].min(), decimals=decimals)}


    def drawClustersOnMap(self, df, kind, photopath='../data/limaMap.png'):
        assert (kind=='source') or (kind=='destination')
        assert os.path.exists(photopath)

        print("\tGenerating Cluster Map for "+str(kind)+" coordinates...")
        clusters = len(df[str(kind)+'Cluster'].unique())
        x = np.arange(clusters+1)                         # used for different colors of clusters
        ys = [i+x+(i*x)**2 for i in range(clusters+1)]
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(ys)))


        ruh_m = plt.imread(photopath)
        BBox = ((df[str(kind)+'_longitude'].min(),
                 df[str(kind)+'_longitude'].max(),
                 df[str(kind)+'_latitude'].min(),
                 df[str(kind)+'_latitude'].max()))

        fig, ax = plt.subplots(figsize = (8,7))

        # For all Clusters
        num=0
        for k in df[str(kind)+'Cluster'].unique():
            ax.scatter(df[df[str(kind)+'Cluster']==k][str(kind)+'_longitude'],
                       df[df[str(kind)+'Cluster']==k][str(kind)+'_latitude'],
                       zorder=1, alpha=0.2, c=[np.random.rand(3,)], s=10)                  # random color
        ax.set_title('Plotting Spatial Data on Lima Map')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
        fig.savefig(os.path.join(self.exportpath,'Clustermap_'+str(kind)+'.png'))
        plt.close(fig)
