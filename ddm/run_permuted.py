from clfw.mnist import PermutedMnist
import numpy as np
import tensorflow as tf

from ddm.alg import vcl, coreset, utils
from generator import Generator


def main():
    hidden_size = [100, 100]
    batch_size = 256
    no_epochs = 100
    single_head = True
    num_tasks = 5

    # Run vanilla VCL
    tf.compat.v1.set_random_seed(12)
    np.random.seed(1)

    coreset_size = 0
    data_gen = Generator(PermutedMnist(num_tasks))

    vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    print(vcl_result)

    # # Run random coreset VCL
    # tf.reset_default_graph()
    # tf.set_random_seed(12)
    # np.random.seed(1)
    #
    # coreset_size = 200
    # data_gen = PermutedMnist(num_tasks)
    # rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    #     coreset.rand_from_batch, coreset_size, batch_size, single_head)
    # print(rand_vcl_result)
    #
    # # Run k-center coreset VCL
    # tf.reset_default_graph()
    # tf.set_random_seed(12)
    # np.random.seed(1)
    #
    # data_gen = PermutedMnistGenerator(num_tasks)
    # kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    #     coreset.k_center, coreset_size, batch_size, single_head)
    # print(kcen_vcl_result)

    # Plot average accuracy
    vcl_avg = np.nanmean(vcl_result, 1)
    # rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
    # kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
    # utils.plot('results/permuted.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)


if __name__ == '__main__':
    main()
