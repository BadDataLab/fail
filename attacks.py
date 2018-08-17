

def jsma_craft(surrogate_model, source_samples, target_info):
    import tensorflow as tf
    import numpy as np
    from dataset import Data
    from cleverhans.attacks import SaliencyMapMethod



    with tf.variable_scope(surrogate_model.scope):
        sess = tf.get_default_session()

        jsma = SaliencyMapMethod(surrogate_model.tf(), back='tf', sess=sess)


        one_hot_target = np.zeros((1, 10), dtype=np.float32)
        one_hot_target[0, target_info['target_class']] = 1
        jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': one_hot_target}

        # Loop over the samples we want to perturb into adversarial examples

        adv_xs = []
        for sample_ind in range(len(source_samples)):
            sample = source_samples.getx(sample_ind)

            adv_xs.append(jsma.generate_np(sample, **jsma_params))
          
            # TODO remove break
            if sample_ind == 2:
              break

        adv_ys = np.concatenate([one_hot_target]*len(adv_xs))

        adv_xs_d = Data(np.concatenate(adv_xs), adv_ys)

        return source_samples, adv_xs_d



def stingray_craft(surrogate_model, source_samples, target_info):
    import numpy as np
    from sklearn.metrics.pairwise import manhattan_distances
    from scipy.sparse import csc_matrix
    from scipy.sparse import vstack
    import sys

    return source_samples,source_samples
    
    class StingRay():
        def __init__(self,config):
            self.config = config
    
        def _get_distance_to_instances(self,target_feature_vector,features):
            distance_to_instances = manhattan_distances(target_feature_vector, features).tolist()[0]
            return np.array(distance_to_instances)
    
        def _get_distance_to_training_instances(self,target_feature_vector,target_label,dataset):
            candidate_base_instances_indices = self._get_candidate_base_instances(dataset.train_labels,target_label)
            selected_available_instances_indices = range(dataset.train_labels.shape[0])
            candidate_selected_available_instances_indices = np.array(selected_available_instances_indices)[candidate_base_instances_indices]
            
            distance_to_training_insts = self._get_distance_to_instances(target_feature_vector,dataset.train_features)
            distance_to_candidate_base_instances = distance_to_training_insts[candidate_base_instances_indices]
            distance_to_candidate_base_instances = zip(candidate_selected_available_instances_indices,distance_to_candidate_base_instances)
            sorted_distance_to_candidate_base_instances = sorted(distance_to_candidate_base_instances,key=lambda e:e[1])
            return sorted_distance_to_candidate_base_instances
    
    
    
        def _merge_crafted_dataset(self, dataset, crafted_feature_matrix, crafted_labels):
            crafted_feature_matrix = csc_matrix(crafted_feature_matrix)
            new_features = vstack((dataset.train_features,crafted_feature_matrix))
            new_labels = np.asarray(np.concatenate((dataset.train_labels, crafted_labels)))
        
            dataset.set_training_set(new_features, new_labels)
            return dataset
    
    
        def _craft_sample(self,target_feature_vector, target_label, base_instance_feature_vector, num_added_features=1,modifiable_feats=None):
            target_feature_vector_dense = np.ravel(target_feature_vector.todense())
            active_target_features = np.where(target_feature_vector_dense == 1)[0]
            num_features = target_feature_vector_dense.shape[0]
        
            base_instance_features = np.ravel(base_instance_feature_vector.todense())
            
            existing_features = np.where(base_instance_features == 1)[0]
            missing_features = filter(lambda e: e not in existing_features,active_target_features)
        
            if modifiable_feats is not None:
                missing_features = filter(lambda e: e in modifiable_feats,missing_features)
        
            if len(missing_features) > 0:
                added_features = np.random.choice(missing_features,min(num_added_features,len(missing_features)),replace=False)
            else:
                added_features = existing_features
        
        
            crafted_feature_vector = np.zeros(num_features, dtype=float)
            crafted_feature_vector[existing_features] = 1
            crafted_feature_vector[added_features] = 1
    
            crafted_feature_vector = np.asarray([crafted_feature_vector])
            crafted_label = np.asarray([1-target_label])
        
            return crafted_feature_vector,crafted_label
    
    
        def _get_candidate_base_instances(self,training_labels,target_label):
            return np.where(training_labels!=target_label)[0]
    
    
        def run(self,target_feature_vector,target_label):            
            dataset = self.config.classifier.dataset
            sorted_distance_to_candidate_base_instances = self._get_distance_to_training_instances(target_feature_vector,target_label,dataset)
                
            newdataset = dataset.copy()
    
            num_crafted_instances = 0
            num_poison_instance = 0
            prediction = target_label
    
            while num_crafted_instances < self.config.max_num_crafted_instances and prediction == target_label:
                sorted_distsances_instance_pick_index = np.random.choice(xrange(len(sorted_distance_to_candidate_base_instances[:self.config.num_nearest_base_instances])),size=1,replace=False)[0]  
                base_instance_pick_index,dist_to_target = sorted_distance_to_candidate_base_instances[sorted_distsances_instance_pick_index]
                
                base_instance_feature_vector = dataset.train_features[base_instance_pick_index]
    
                crafted_instances_feature_vectors, crafted_instances_labels = self._craft_sample(target_feature_vector, target_label, base_instance_feature_vector, num_added_features=self.config.num_added_features,modifiable_feats=self.config.readonly_features)
    
                num_crafted_instances += 1
    
                if num_crafted_instances % 10 == 0:
                    print ('crafted:',num_crafted_instances,'/',self.config.max_num_crafted_instances)
                    sys.stdout.flush()
    
    
                newdataset = self._merge_crafted_dataset(newdataset, crafted_instances_feature_vectors, crafted_instances_labels)
                    
                num_poison_instance += 1
                if num_poison_instance % 50 == 0:
                    print ('Crafted ... ',num_crafted_instances)
    
                    newclf = self.config.classifier.new(newdataset)
                    newclf.train()
                    prediction = newclf.predict(target_feature_vector)[0]
            
            print ('Done Targeting 1. Crafted ', num_poison_instance)
            if prediction == target_label:
                print ('unscessful')
            return newdataset


