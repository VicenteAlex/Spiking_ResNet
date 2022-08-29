
if __name__ == '__main__':
    import torchvision
    from   torch.utils.data.dataloader import DataLoader
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import argparse
    import numpy as np
    from utils import *
    import time
    from model import *
    import os

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Test S-ResNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--data_folder',           default='data', type=str, help='Folder for saving data')
    parser.add_argument('--arch',                  default='sresnet', type=str, help='architecture used by the model')
    parser.add_argument('--n',                     default=6, type=int, help='Depth scaling of the S-ResNet')
    parser.add_argument('--nFilters',              default=32, type=int, help='Width scaling of the S-ResNet')
    parser.add_argument('--boosting',              default=False, action='store_true', help='Use boosting layer')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset [cifar10, cifar100, cifar10dvs]')
    parser.add_argument('--batch_size',            default=500,       type=int,   help='Batch size')
    parser.add_argument('--num_steps',             default=50,    type=int, help='Number of time-step')
    parser.add_argument('--leak_mem', default=0.874, type=float, help='Leak_mem')
    parser.add_argument('--device', default=None, type=int, help='gpu number to use')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--poisson_gen',default=False, action='store_true', help='use poisson spike generation')

    global args
    args = parser.parse_args()

    if args.device is not None:
        torch.cuda.set_device(args.device)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define model and dataset

    leak_mem = args.leak_mem
    batch_size      = args.batch_size
    num_steps       = args.num_steps

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        num_cls = 10
        img_size = 32

        test_set = torchvision.datasets.CIFAR10(root=args.data_folder, train=False,
                                                download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        num_cls = 100
        img_size = 32

        test_set = torchvision.datasets.CIFAR100(root=args.data_folder, train=False,
                                                download=True, transform=transform_test)

    elif args.dataset == 'cifar10dvs':
        num_cls = 10
        img_size = 64

        split_by = 'number'
        normalization = None
        T = args.num_steps  # number of frames

        dataset_dir = os.path.join(args.data_folder, args.dataset)
        if os.path.isdir(dataset_dir) is not True:
            os.mkdir(dataset_dir)

        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        # from spikingjelly.datasets import split_to_train_test_set  # Original function

        # Redefining split function to make it faster
        def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int,
                                    random_split: bool = False):
            '''
            :param train_ratio: split the ratio of the origin dataset as the train set
            :type train_ratio: float
            :param origin_dataset: the origin dataset
            :type origin_dataset: torch.utils.data.Dataset
            :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
            :type num_classes: int
            :param random_split: If ``False``, the front ratio of samples in each classes will
                    be included in train set, while the reset will be included in test set.
                    If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
                    ``numpy.randon.seed``
            :type random_split: int
            :return: a tuple ``(train_set, test_set)``
            :rtype: tuple
            '''
            import math
            label_idx = []

            if len(origin_dataset.samples) != 10000:  # If number of samples has been modified store label one by one
                for i in range(num_classes):
                    label_idx.append([])
                for i, item in enumerate(origin_dataset):
                    y = item[1]
                    if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
                        y = y.item()
                    label_idx[y].append(i)
            else:
                for i in range(10):  # Else, 1000 images per class
                    label_idx.append(list(range(i * 1000, (i + 1) * 1000)))
            train_idx = []
            test_idx = []
            if random_split:
                for i in range(num_classes):
                    np.random.shuffle(label_idx[i])

            for i in range(num_classes):
                pos = math.ceil(label_idx[i].__len__() * train_ratio)
                train_idx.extend(label_idx[i][0: pos])
                test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

            return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

        origin_set = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T,
                                split_by=split_by)
        _, test_set = split_to_train_test_set(0.9, origin_set, 10)

    else:
        print("Dataset name not found")
        exit()

    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # Instantiate the SNN model and optimizer

    if args.arch == 'sresnet':
        model = SResnet(n=args.n, nFilters=args.nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls,
                        boosting=args.boosting, poisson_gen=args.poisson_gen)
    elif args.arch == 'sresnet_nm':
        model = SResnetNM(n=args.n, nFilters=args.nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)

    else:
        print("Architecture name not found")
        exit()

    # Load weigths

    model_dict = torch.load(args.model_path, map_location='cpu')
    state_dict = model_dict['state_dict']
    reload_epoch = model_dict['global_step']
    best_acc = model_dict['accuracy']
    own_state = model.state_dict()

    original_steps = 50

    for name, param in state_dict.items():

        if name in own_state.keys():
            print(name)
            own_state[name].copy_(param)
        else:
            print('skiping: ' + name)

    print("Reloaded weigths, checkpoint taken at epoch {reload_epoch} with validation "
          "accuracy {best_acc} ".format(reload_epoch=reload_epoch,best_acc=best_acc))

    ###############

    model.eval()  # test mode
    if args.device is not None:
        model.cuda()

    # Compute inference in all test set

    acc_top1, acc_top5 = [], []
    time_vec = []
    with torch.no_grad():

        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.to(args.device, dtype=torch.float)
            labels = labels.cuda()

            start_time = time.time()
            out = model(inputs)
            time_elapsed = time.time() - start_time
            time_vec.append(time_elapsed)
            print ('Time elapsed: '+ str(time_elapsed))

            prec1, prec5 = accuracy(out, labels, topk=(1, 5))
            acc_top1.append(float(prec1))
            acc_top5.append(float(prec5))

    test_accuracy = np.mean(acc_top1)
    test_accuracy_top5 = np.mean(acc_top5)

    print("test_accuracy : {}".format(test_accuracy))
    print("test_accuracy top 5 : {}".format(test_accuracy_top5))
    print("Average time per batch: {}".format(np.mean(time_vec)))