import numpy as np
import unittest
from data_util import *
from explain_pipeline import *
from model import _CNN


class TestDataUtil(unittest.TestCase):

    def test_split_csv(self):
        split_csv("ADNI3/ADNI3.csv", random_seed=20)

    def test_CNN_Data(self):
        data = CNN_Data('ADNI3/test_data.csv')
        print(data)

    def test_read_csv(self):
        file_ls, ad_ls = read_csv(MRI_FOLDER)
        print(file_ls)
        print(ad_ls)


class TestExplainPipeline(unittest.TestCase):

    def setUp(self) -> None:
        global myModel
        myModel = _CNN(20, 0.1)
        myModel_dict = torch.load("ADNI3/model_para/cnn_best.pth", map_location=torch.device('cpu'))
        myModel.load_state_dict(myModel_dict["state_dict"])

    def test_prepare_dataloaders(self):
        bg_loader, test_loader = prepare_dataloaders('ADNI3/bg_data.csv', 'ADNI3/test_data.csv')
        correct = 0  # record the correct number
        # record if every mri is detected correctly
        name_ls = []
        pre_ls = []
        act_ls = []
        for idx, (mri, name, ad) in enumerate(test_loader, 1):
            pre_ad = myModel(mri)
            pre_ad = pre_ad.max(dim=1, keepdim=False)[1]
            correct += pre_ad.eq(ad).sum().item()
            name_ls.append(name[0])
            pre_ls.append(pre_ad.item())
            act_ls.append(ad.item())
        df = pd.DataFrame(data={'filename': name_ls, 'predict': pre_ls, 'acture': act_ls})
        df.to_csv("ADNI3/Forecast.csv", index=False)
        print(f'correctly predicated:{correct}')

    def test_create_SHAP_values(self):
        create_SHAP_values(bg_loader, test_loader, 6, "ADNI3/shape_values")

    def test_plot_shap_on_mri(self):
        plot_shap_on_mri(
            'ADNI3/data/ADNI_135_S_6545_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190913133638192_107_S873086_I1226543.npy',
            'ADNI3/shape_values/ADNI_135_S_6545_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190913133638192_107_S873086_I1226543.npy',
            'exist_AD'
        )

        # Non exist AD
        plot_shap_on_mri(
            'ADNI3/data/ADNI_099_S_6038_MR_Accelerated_Sag_IR-FSPGR___br_raw_20181112140012329_32_S746377_I1071981.npy',
            'ADNI3/shape_values/ADNI_099_S_6038_MR_Accelerated_Sag_IR-FSPGR___br_raw_20181112140012329_32_S746377_I1071981.npy',
            'non_exist_AD'
        )

    def test_output_top_10_lst(self):
        output_top_10_lst(r'ADNI3/top_10.csv')
