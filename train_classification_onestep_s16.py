import os
import sys
import torch
import numpy as np
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from models.dgcnn_cls import *
from models.dgcnn_utils import cal_loss
# from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=16, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--inner_epoch', default=200, type=int, help='number of epoch in inner training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_assemble', action='store_true', default=False, help='use assemble training')
    parser.add_argument('--angles', type=float, default=None, help='random rotation bound')
    parser.add_argument('--scales', type=float, default=None, help='random scale bound')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help='use assemble training')
    parser.add_argument('--step_size', default=0.01, type=float, help='attack step size')
    parser.add_argument('--iters', default=0, type=int, help='attack steps')
    parser.add_argument('--aw', action='store_true', default=False, help='axis wise attack')
    parser.add_argument('--rp', action='store_true', default=False, help='rotation pool')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(classifier, loader, rotation_pool, num_class):
    args = parse_args()
    mean_correct = []
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        points = points.data.numpy()
        # points = provider.random_point_dropout(points)
        # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3], _ = provider.random_sr_point_cloud(points[:, :, 0:3])
        if not args.rp:
            points, _ = provider.random_rotate_point_cloud(points, args.angles)
        else:
            sample_i = 0
            for category in target:
                rotate = random.sample(rotation_pool[category.item()], 1)[0]
                R = provider.generate_a_rotate_matrix(rotate)
                points[sample_i, :, 0:3] = np.dot(points[sample_i, :, 0:3], R)
                sample_i += 1
        # print(sample_i)
        points = torch.from_numpy(points)
        # pred, trans_feat = classifier_train(points_adv.transpose(2, 1))

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points.transpose(2, 1))
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)


    return instance_acc, class_acc


class CWFGSM(object):
    def __init__(self, iters, step_size, angles, ax_wise=False):
        self.iters = iters
        self.angles = angles
        # self.scales = args.scales
        self.step_size = step_size
        self.ax_wise = ax_wise

    def forward(self, target_cls, points, labels):
        B = points.shape[0]
        iters = self.iters
        step_size = self.step_size
        if not self.ax_wise:
            # points = points.data.cpu().numpy()
            trans = self.angles * (2 * np.random.rand(3, B) - 1)
            # trans = np.ones((3, B)) - 0.5
            # trans1 = copy.deepcopy(trans)
            # points = torch.from_numpy(points)
            # points = torch.from_numpy(points.data.cpu().numpy())
            points_adv = points.detach().clone()
            delta = torch.from_numpy(trans)
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=step_size)

            for i in range(iters):
                with torch.no_grad():
                    delta.clamp_(-self.angles, self.angles)
                for j in range(B):
                    r_angles = delta[:, j] * math.pi
                    c0, c1, c2 = torch.cos(r_angles[0]), torch.cos(r_angles[1]), torch.cos(r_angles[2])
                    s0, s1, s2 = torch.sin(r_angles[0]), torch.sin(r_angles[1]), torch.sin(r_angles[2])
                    f1, f2, f3 = c2 * c1, c2 * s1 * s0 - s2 * c0, c2 * s1 * c0 + s2 * s0
                    f4, f5, f6 = s2 * c1, s2 * s1 * s0 + c2 * c0, s2 * s1 * c0 - c2 * s0
                    f7, f8, f9 = -s1, c1 * s0, c0 * c1
                    points_adv[j, :, 0] = f1 * points[j, :, 0] + f4 * points[j, :, 1] + f7 * points[j, :, 2]
                    points_adv[j, :, 1] = f2 * points[j, :, 0] + f5 * points[j, :, 1] + f8 * points[j, :, 2]
                    points_adv[j, :, 2] = f3 * points[j, :, 0] + f6 * points[j, :, 1] + f9 * points[j, :, 2]
                points_adv = points_adv.cuda()
                outputs, trans_feat = target_cls(points_adv.permute(0, 2, 1))
                # loss = criterion(outputs, labels.long(), trans_feat)
                loss = torch.sum(self._f(outputs, labels.long()))
                # print(loss)
                # print(loss, i)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # with torch.no_grad():
                #     delta.clamp_(-self.angles, self.angles)
                # print(delta)

            adversarial_examples = points_adv
            delta = delta.detach().numpy()
        else:
            trans = self.angles * (2 * np.random.rand(3, B) - 1)
            # trans = np.ones((3, B)) - 0.5
            points_adv = points.detach().clone()
            delta1 = torch.from_numpy(trans[0])
            delta2 = torch.from_numpy(trans[1])
            delta3 = torch.from_numpy(trans[2])
            delta1 = delta1.cuda()
            delta2 = delta2.cuda()
            delta3 = delta3.cuda()
            delta1.requires_grad = True
            delta2.requires_grad = True
            delta3.requires_grad = True
            optimizer1 = torch.optim.Adam([delta1], lr=step_size)
            optimizer2 = torch.optim.Adam([delta2], lr=step_size)
            optimizer3 = torch.optim.Adam([delta3], lr=step_size)
            for i in range(iters):
                with torch.no_grad():
                    delta1.clamp_(-self.angles, self.angles)
                    delta2.clamp_(-self.angles, self.angles)
                    delta3.clamp_(-self.angles, self.angles)
                c0, c1, c2 = torch.cos(delta1 * math.pi), torch.cos(delta2 * math.pi), torch.cos(delta3 * math.pi)
                s0, s1, s2 = torch.sin(delta1 * math.pi), torch.sin(delta2 * math.pi), torch.sin(delta3 * math.pi)
                f1, f2, f3 = c2 * c1, c2 * s1 * s0 - s2 * c0, c2 * s1 * c0 + s2 * s0
                f4, f5, f6 = s2 * c1, s2 * s1 * s0 + c2 * c0, s2 * s1 * c0 - c2 * s0
                f7, f8, f9 = -s1, c1 * s0, c0 * c1
                # print(points_adv[0, :, 0], f1, points[0, :, 0])
                points_adv[0, :, 0] = f1 * points[0, :, 0] + f4 * points[0, :, 1] + f7 * points[0, :, 2]
                points_adv[0, :, 1] = f2 * points[0, :, 0] + f5 * points[0, :, 1] + f8 * points[0, :, 2]
                points_adv[0, :, 2] = f3 * points[0, :, 0] + f6 * points[0, :, 1] + f9 * points[0, :, 2]
                points_adv = points_adv.cuda()
                points_adv_t = points_adv.permute(0, 2, 1)
                outputs, trans_feat = target_cls(points_adv_t)
                loss = torch.sum(self._f(outputs, labels.long()))
                # print(loss)
                gradients = torch.autograd.grad(loss, points_adv_t, retain_graph=True)[0]
                gradients = gradients.permute(0, 2, 1)
                delta_x, x = gradients[:, :, 0], points_adv[:, :, 0]
                delta_y, y = gradients[:, :, 1], points_adv[:, :, 1]
                delta_z, z = gradients[:, :, 2], points_adv[:, :, 2]
                # 1. angles along three axis
                Lphi_x = torch.sum((-z) * delta_y + y * delta_z, dim=1)
                Lphi_y = torch.sum((-x) * delta_z + z * delta_x, dim=1)
                Lphi_z = torch.sum((-y) * delta_x + x * delta_y, dim=1)
                Lphi = torch.cat([Lphi_x.view(B, 1), Lphi_y.view(B, 1), Lphi_z.view(B, 1)], dim=1)
                max_axis_id = torch.argmax(torch.abs(Lphi), dim=1)
                if max_axis_id[0] == 0:
                    optimizer1.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer1.step()
                elif max_axis_id[0] == 1:
                    optimizer2.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer2.step()
                else:
                    optimizer3.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer3.step()
            delta = torch.cat([delta1, delta2, delta3])
            delta = delta.detach().cpu().numpy()
            # print(delta.shape)

            adversarial_examples = points_adv
        return adversarial_examples, np.reshape(delta, (3, B))

    def _f(self, outputs, labels):
        # sm = torch.nn.Softmax(dim=1)
        # outputs = -torch.log(sm(outputs))
        outputs = -outputs
        y_onehot = torch.zeros_like(outputs).scatter(1, labels.view(-1, 1), 1)
        real = (y_onehot * outputs).sum(dim=1)
        other, _ = torch.max((1 - y_onehot) * outputs, dim=1)
        loss = other - real
        return loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classifications16')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/pointnet_cls.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = '../../../data/modelnet40_normal_resampled/'
    # train_dataset = ModelNetDataLoader(root=data_path, npoints=1024, num_category=40, split='train')
    # test_dataset = ModelNetDataLoader(root=data_path, npoints=1024, num_category=40, split='test')
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                                               num_workers=16, drop_last=True)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                              num_workers=16)
    # new data_sets!!!!
    train_points = np.load('train1024points16_2.npy')
    train_labels = np.load('train1024labels16_2.npy').flatten()
    print('train data size:', train_labels.shape)
    test_points = np.load('test1024points16_2.npy')
    test_labels = np.load('test1024labels16_2.npy').flatten()
    print('test data size:', test_labels.shape)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_points), torch.from_numpy(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_points), torch.from_numpy(test_labels))

    # train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_points[0:50, :, :]), torch.from_numpy(train_labels[0:50]))
    # test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_points), torch.from_numpy(test_labels))
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=17, shuffle=False, num_workers=16)

    '''TRAIN MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module('pointnet_cls')
    shutil.copy('train_classification_dynamic.py', str(exp_dir))
    shutil.copy('provider.py', str(exp_dir))
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    classifier_train = model.get_model(num_class, normal_channel=args.use_normals)
    classifier_train = classifier_train.cuda()
    criterion = model.get_loss()
    criterion = criterion.cuda()
    pretrained_dir = Path('./pretrained_models/shapenet16/pn1.pth')
    checkpoint_pre = torch.load(pretrained_dir)
    classifier_train.load_state_dict(checkpoint_pre['model_state_dict'])
    log_string('Use Pretrained trainModel pn1')


    '''EVAL MODEL LOADING'''
    model_eval1 = importlib.import_module('pointnet_cls')
    classifier_eval1 = model_eval1.get_model(num_class, normal_channel=args.use_normals)
    classifier_eval1 = classifier_eval1.cuda()
    # criterion1 = model_eval1.get_loss()
    # criterion1 = criterion1.cuda()
    pretrained_dir = Path('./pretrained_models/shapenet16/pn1.pth')
    checkpoint_pre = torch.load(pretrained_dir)
    classifier_eval1.load_state_dict(checkpoint_pre['model_state_dict'])
    log_string('Use pretrain evalmodel pn1')

    model_eval2 = importlib.import_module('pointnet2_cls_ssg')
    classifier_eval2 = model_eval2.get_model(num_class, normal_channel=args.use_normals)
    classifier_eval2 = classifier_eval2.cuda()
    # criterion2 = model_eval2.get_loss()
    # criterion2 = criterion2.cuda()
    pretrained_dir = Path('./pretrained_models/shapenet16/pn2.pth')
    checkpoint_pre = torch.load(pretrained_dir)
    classifier_eval2.load_state_dict(checkpoint_pre['model_state_dict'])
    log_string('Use pretrain evalmodel pn2')

    classifier_eval3 = DGCNN(num_class)
    classifier_eval3 = classifier_eval3.cuda()
    classifier_eval3 = nn.DataParallel(classifier_eval3)
    # criterion3 = cal_loss()
    pretrained_dir = Path('./pretrained_models/shapenet16/dgcnn.t7')
    classifier_eval3.load_state_dict(torch.load(pretrained_dir))
    log_string('Use pretrain evalmodel dgcnn')

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    attack = CWFGSM(args.iters, args.step_size, args.angles, args.aw)
    '''TRANING'''
    logger.info('Start training...')

    log_string('One-Step Optimization')
    adv_samples = []
    adv_labels = []
    rotation_pool = {}
    attack_num = 0
    attack_correct = 0
    log_string('eval_pn1:{}'.format(classifier_eval1.state_dict()['fc1.weight']))
    log_string('eval_pn2:{}'.format(classifier_eval2.state_dict()['fc1.weight']))
    log_string('eval_dgcnn:{}'.format(classifier_eval3.state_dict()['module.linear1.weight']))

    log_string('train:{}'.format(classifier_train.state_dict()['fc1.weight']))


    classifier_train = classifier_train.train()
    # max to find most aggressive
    log_string('Max Step to Find Most Aggressive')
    #for classifier_eval in [classifier_eval1, classifier_eval2, classifier_eval3]:
    for classifier_eval in [classifier_eval1, classifier_eval3]:
        for batch_id_max, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # print(points.shape, target.shape)
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            adv_points, trans = attack.forward(classifier_eval.eval(), points, target)
            if not args.rp:
                adv_samples.append(adv_points)
                adv_labels.append(target)
            else:
                sample_i = 0
                for category in target.data.cpu().numpy():
                    # category = str(category)
                    # print(category)
                    if category in rotation_pool:
                        # print(rotation_pool)
                        # print(rotation_pool[category])
                        rotation_pool[category].append(trans[:, sample_i])
                        # all_trans_start[label].append(trans_start[:, sample_i])
                    else:
                        rotation_pool[category] = [trans[:, sample_i]]
                    sample_i += 1
                        # all_trans_start[label] = [trans_start[:, i]]
            outputs, _ = classifier_eval(adv_points.permute(0, 2, 1))
            # points = points.permute(0,2,1)
            # outputs, _ = self.target_cls(points)
            pred_choice = outputs.data.max(1)[1]
            correct_num = pred_choice.eq(target.long().data).cpu().sum()
            # _, predicted = torch.max(outputs, 1)
            attack_correct += correct_num
            attack_num += target.size(0)
            acc = attack_correct / float(attack_num)
            log_string('Acc after Attack: %.4f' % acc)
        # log_string('Acc after Attack: %.4f' % acc)

    if not args.rp:
        adv_samples = torch.cat(adv_samples).data.cpu().numpy()
        adv_labels = torch.cat(adv_labels).data.cpu().numpy()
        print(adv_samples.shape, adv_labels.shape)
        log_string(True in np.isnan(adv_samples))
        log_string(True in np.isnan(adv_labels))
        adv_dataset = torch.utils.data.TensorDataset(torch.from_numpy(adv_samples), torch.from_numpy(adv_labels))  # create your datset
        adv_dataloader = torch.utils.data.DataLoader(adv_dataset, batch_size=17, shuffle=True)  # create your dataloader
        logger.info('End of Min Step......')
    else:
        # for ts in rotation_pool.values():
        #     print(len(ts))
        #     for t in ts:
        #         log_string(True in np.isnan(t))
        adv_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=17, shuffle=True,
                                                      num_workers=16, drop_last=True)

    if args.angles:
        print('rotate data with %.2f' % args.angles)
    optimizer = torch.optim.Adam(
        classifier_train.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    log_string('begin optim: {}'.format(optimizer.param_groups[0]['lr']))
    # min steps
    log_string('Min Step to Optimize on Most Aggressive')
    for epoch_adv in range(args.inner_epoch):
        log_string('Epoch %d (%d/%s):' % (epoch_adv + 1, epoch_adv + 1, args.inner_epoch))
        mean_correct = []
        for batch_id_min, (points_adv, target_adv) in tqdm(enumerate(adv_dataloader, 0), total=len(adv_dataloader),
                                               smoothing=0.9):
            if not args.use_cpu:
                points_adv, target_adv = points_adv.cuda(), target_adv.cuda()
            if not args.rp:
                pred, trans_feat = classifier_train(points_adv.transpose(2, 1))
            else:
                points_adv = points_adv.data.cpu().numpy()
                sample_i = 0
                for category in target_adv:
                    rotate = random.sample(rotation_pool[category.item()], 1)[0]
                    R = provider.generate_a_rotate_matrix(rotate)
                    points_adv[sample_i, :, 0:3] = np.dot(points_adv[sample_i, :, 0:3], R)
                    sample_i += 1
                # print(sample_i)
                points_adv = torch.from_numpy(points_adv)
                points_adv = points_adv.cuda()
                pred, trans_feat = classifier_train(points_adv.transpose(2, 1))
            # log_string('cloud:{}'.format(points_adv))
            # log_string('target:{}'.format(target_adv))
            loss = criterion(pred, target_adv.long(), trans_feat)
            # log_string('loss:{}'.format(loss))
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target_adv.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points_adv.size()[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
        scheduler.step()
        log_string('One-Step {} optim: {}'.format(epoch_adv, optimizer.param_groups[0]['lr']))
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        test_acc_mean_ins = []
        test_acc_mean_cls = []

        for i in range(1):
            instance_acc_single, class_acc_single = test(classifier_train.eval(), testDataLoader,
                                                         rotation_pool, num_class=num_class)
            log_string(
                'Current index %d, Test Instance Accuracy: %f, Class Accuracy: %f' % (
                i, instance_acc_single, class_acc_single))
            test_acc_mean_ins.append(instance_acc_single)
            test_acc_mean_cls.append(class_acc_single)

        instance_acc = np.mean(test_acc_mean_ins)
        class_acc = np.mean(test_acc_mean_cls)

        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        if (instance_acc >= best_instance_acc):
            best_instance_acc = instance_acc
            best_epoch = epoch_adv + 1
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier_train.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

    global_epoch += 1
    logger.info('End of One Max Step %d...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
    # path = '../../../data/modelnet40_normal_resampled/modelnet40_train_1024pts.dat'
