import numpy as np
from numpy import linalg as LA

import pdb

class Image_Scoring:
    """
    Evaluate distance between diff images
    """
    def __init__(self):
        pass

    def cosine_distance(self,query_array,db_array):
        """
        Find cosine-distance

        Example:
            query_array = np.array([[1,2],[1,1]])
            db_array = np.array([2,4],[1,1],[1,3])

        :param query_array: query images(N1xM array)
        :param db_array: database images(N2xM array)
        :return: cosine distance(N1xN2 array)
        """

        #pdb.set_trace()

        norm_query = LA.norm(query_array,axis=1)
        norm_query = norm_query.reshape(len(norm_query),1)

        norm_db = LA.norm(db_array,axis=1)
        norm_db = norm_db.reshape(1,len(norm_db))

        dot_query_db = np.dot(query_array,db_array.T).astype(float)

        cosin_dist = dot_query_db/(np.dot(norm_query,norm_db))

        return cosin_dist


if __name__ == '__main__':
    scorer = Image_Scoring()

    query = np.array([[1,2],[1,1]])

    db = np.array([[2,4],[1,1],[1,3]])

    dis = scorer.cosine_distance(query,db)

    pdb.set_trace()

    pass