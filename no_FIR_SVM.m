% SVM
clear; close all; clc;

%% Step 1: 데이터 로드 및 초기화
desktop_path = fullfile(getenv('USERPROFILE'), 'Desktop');
base_folder = fullfile(desktop_path, 'instu_data');

instrument_folders = {'gac', 'pia', 'sax', 'vio'};  
labels = {'gac', 'pia', 'sax', 'vio'};

all_features = [];
all_labels = [];

%% Step 2: 특징 추출 (FIR 필터 제거)
for i = 1:length(instrument_folders)
    folder_path = fullfile(base_folder, instrument_folders{i});
    audio_files = dir(fullfile(folder_path, '*.wav'));  

    local_features = [];
    local_labels = [];
    for j = 1:length(audio_files)
        file_path = fullfile(folder_path, audio_files(j).name);
        [audio, Fs] = audioread(file_path);

        % 오디오 데이터를 정규화
        audio = audio / max(abs(audio));

        % FIR 필터 제거: 원본 오디오 사용
        processed_audio = audio;

        % 특징 추출 (MFCC, RMS, ZCR, Spectral Flatness, Crest Factor, Spectral Centroid)
        window_size = 1024;  
        hop_size = 512;      
        analysis_window = hamming(window_size, 'periodic');  
        mfcc_features = mfcc(processed_audio, Fs, 'Window', analysis_window, ...
                              'OverlapLength', window_size - hop_size);
        rms_energy = sqrt(mean(processed_audio .^ 2));  
        zcr = sum(abs(diff(sign(processed_audio)))) / length(processed_audio);  
        spectral_flatness = geo_mean(abs(processed_audio)) / mean(abs(processed_audio));  
        crest_factor = max(abs(processed_audio)) / rms_energy;  
        spectral_centroid = sum((1:length(processed_audio))' .* abs(processed_audio)) / sum(abs(processed_audio));  
        spectral_bandwidth = sqrt(sum(((1:length(processed_audio))' - spectral_centroid) .^ 2 .* abs(processed_audio)) / sum(abs(processed_audio)));  

        % 모든 특징 병합
        feature_vector = [mean(mfcc_features, 1), rms_energy, zcr, spectral_flatness, crest_factor, ...
                          spectral_centroid, spectral_bandwidth];
        local_features = [local_features; feature_vector];
        local_labels = [local_labels; labels(i)];
    end

    all_features = [all_features; local_features];
    all_labels = [all_labels; local_labels];
end
disp('FIR 필터 제거 후 특징 추출 완료.');

%% Step 3: 데이터 정규화 및 분리
feature_min = min(all_features, [], 1);
feature_max = max(all_features, [], 1);
scaled_features = (all_features - feature_min) ./ (feature_max - feature_min);

num_samples = size(scaled_features, 1);
random_idx = randperm(num_samples);

train_ratio = 0.8;
train_idx = random_idx(1:round(train_ratio * num_samples));
test_idx = random_idx(round(train_ratio * num_samples) + 1:end);

train_features = scaled_features(train_idx, :);
test_features = scaled_features(test_idx, :);

train_labels = categorical(all_labels(train_idx));
test_labels = categorical(all_labels(test_idx));

disp('데이터 정규화 및 분리 완료.');

%% Step 4: SVM 모델 학습
class_weights = 1 ./ countcats(train_labels);
weights = arrayfun(@(x) class_weights(find(categories(train_labels) == x, 1)), train_labels);

template = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);

mdl = fitcecoc(train_features, train_labels, ...
    'Learners', template, ...
    'Weights', weights, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ShowPlots', false, ...
        'Verbose', 1));

disp('SVM 모델 학습 및 최적화 완료.');

%% Step 5: 모델 평가
predicted_labels = predict(mdl, test_features);

accuracy = sum(predicted_labels == test_labels) / length(test_labels) * 100;
disp(['모델 정확도: ', num2str(accuracy), '%']);

confusionchart(test_labels, predicted_labels);
title('Confusion Matrix - SVM Model without FIR Filter');

categories_list = categories(test_labels);
for i = 1:length(categories_list)
    category = categories_list{i};
    idx = test_labels == category;
    category_accuracy = sum(predicted_labels(idx) == test_labels(idx)) / sum(idx) * 100;
    disp(['악기: ', category, ' 정확도: ', num2str(category_accuracy), '%']);
end
