import math
import pandas as pd
import os
from pylab import rcParams
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging

logger = logging.getLogger(__file__)


class ClusterAnalysis():
    """ This class implements the code for clustering the data (source/destination)
    """
    def __init__(self, args, df):
        logger.info("Cluster Analysis")
        self.explore = args.explore
        self.exportpath = os.path.join(args.exportpath, 'Clustering')
        df = self.clean_data(df)
        if explore!=False:
            self.cluster_exploration(df, 'source')
            self.cluster_exploration(df, 'destination')
        df = self.cluster_data(df)
        self.draw_clusters_on_map(df, 'source')
        self.draw_clusters_on_map(df, 'destination')

    def cluster_exploration(self, df_geo, kind):
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
        logger.info("\tExploring K for KMeans for {k} coordinates...".format(k=kind))
        df_geo = df_geo[['{k}_longitude'.format(k=kind), '{k}_longitude'.format(k=kind)]]

        # Explore K
        for k in K:
            kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(df_geo)
            performanceList.append(kmeanModel.inertia_)

        # Plot the elbow
        plt.plot(K, performanceList, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Error vs K in {k} coordinates'.format(k=kinds))
        if not os.path.exists(self.exportpath):
            os.mkdir(outputfolder)
        plt.savefig(os.path.join(self.exportpath,'elbow_rule_{k}.png'.format(k=kind)))
        plt.close()

    def clean_data(self, df):
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

    def cluster_data(self, df, k=15):                                    # default after elbow rule
        """ Run clustering using best known K-parameter
        Args:
            df: the dataframe to apply clustering to
            k: optimal number of clusters for K-Means
        """
        logger.info("\tPerforming K-means for both source and destination coordinates with K={k}".format(k=k))
        kmeans_model = KMeans(n_clusters=k, n_jobs=-1).fit(df[['source_latitude', 'source_longitude']])
        df['sourceCluster'] = kmeans_model.predict(df[['source_latitude', 'source_longitude']])

        kmeans_model = KMeans(n_clusters=k, n_jobs=-1).fit(df[['destination_latitude', 'destination_longitude']])
        df['destinationCluster'] = kmeans_model.predict(df[['destination_latitude', 'destination_longitude']])
        return df

    def calculate_boarders(self, df, decimals=4):
        # Needed to download Lima map from 
        """
        Calculates the boarders of the Lima map in order to drow the clusters in
        Source: https://www.openstreetmap.org/export#map=5/51.500/-0.100
        Args:
            df: the dataframe of coordinates
            decimals: the number of decimals to round to
        """
        sourceDict = {'longitude_max': np.around(df['source_longitude'].max(), decimals=decimals),
                      'longitude_min': np.around(df['source_longitude'].min(), decimals=decimals),
                      'latitude_max': np.around(df['source_latitude'].max(), decimals=decimals),
                      'latitude_min': np.around(df['source_latitude'].min(), decimals=decimals)}

        destinationDict = {'longitude_max': np.around(df['destination_longitude'].max(), decimals=decimals),
                      'longitude_min': np.around(df['destination_longitude'].min(), decimals=decimals),
                      'latitude_max': np.around(df['destination_latitude'].max(), decimals=decimals),
                      'latitude_min': np.around(df['destination_latitude'].min(), decimals=decimals)}


    def draw_clusters_on_map(self, df, kind, photopath='../data/limaMap.png'):
        """
        Draws the clusters on the map (image) and persists it to disk
        Args:
            df: dataframe of coordinates  
            kind: source/destination
        """
        assert (kind=='source') or (kind=='destination')
        assert os.path.exists(photopath)

        logger.info("\tGenerating Cluster Map for {k} coordinates...".format(k=kind))
        clusters = len(df[str(kind)+'Cluster'].unique())
        x = np.arange(clusters+1)                         # used for different colors of clusters
        ys = [i+x+(i*x)**2 for i in range(clusters+1)]
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(ys)))


        ruh_m = plt.imread(photopath)
        BBox = ((df['{k}_longitude'.format(k=kind)].min(),
                 df['{k}_longitude'.format(k=kind)].max(),
                 df['{k}_latitude'.format(k=kind)].min(),
                 df['{k}_latitude'.format(k=kind)].max()))

        fig, ax = plt.subplots(figsize = (8,7))

        # For all Clusters
        num=0
        for k in df[str(kind)+'Cluster'].unique():
            ax.scatter(df[df[str(kind)+'Cluster']==k]['{k}_longitude'.format(k=kind)],
                       df[df[str(kind)+'Cluster']==k]['{k}_latitude'.format(k=kind)],
                       zorder=1, alpha=0.2, c=[np.random.rand(3,)], s=10)                  # random color
        ax.set_title('Plotting Spatial Data on Lima Map')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
        fig.savefig(os.path.join(self.exportpath, 'Clustermap_{k}.png'.format(k=kind)))
        plt.close(fig)
