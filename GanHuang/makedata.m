clc;clear;close all;
EEG=zeros(64,750,40);
for idx_sub=1%:70
    clc;disp(idx_sub)
    data=[];
    load(['..\data\data3\S',num2str(idx_sub),'.mat']);
    [~,label]=sort(data.suppl_info.freqs);
    EEG=EEG+squeeze(mean(data.EEG(:,1:750,:,label),3));
end
EEG=EEG/70;

LW_init();
option.dimension_descriptors={'channels','X','epochs','Y'};
option.unit='amplitude';
option.xunit='time';
option.yunit='frequency';
option.xstart=-0.5;
option.xstep=0.004;
option.ystart=0;
option.ystep=1;
option.is_save=0;
option.filename='EEG';
lwdata=FLW_import_mat.get_lwdata(EEG,option);
for ch_idx=1:64
    lwdata.header.chanlocs(ch_idx).labels=...
        data.suppl_info.chan{ch_idx,4};
end
option=struct('type','channel','items',{{'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'}},'suffix','','is_save',0);
lwdata= FLW_selection.get_lwdata(lwdata,option);
CLW_save(lwdata);




