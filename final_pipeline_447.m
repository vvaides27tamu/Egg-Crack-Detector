clear
clc
close all

originalImage = imread('../ECEN 447/final project images/im4.jpg');
originalImage = double(im2gray(originalImage));
manualAnnotations = im2gray(imread('../ECEN 447/final project images/4DONE.png'));

%threshold annotations
manualAnnotations(manualAnnotations > 128) = 256;
manualAnnotations = edge(manualAnnotations, 'log');

%smooth
n = 3;
mean_filter = 1/n^2 * ones(n);
smoothed = conv2(originalImage, mean_filter, 'same');

%lowpass filter for illumination correction
[M, N] = size(smoothed);
P = 2 * M;
Q = 2 * N;
paddedImage = padarray(smoothed, [P-M, Q-N], 0, 'post');
F = fftshift(fft2(paddedImage));
D0 = 100; %cutoff freq
[u, v] = meshgrid(1:Q, 1:P);
D_uv = sqrt((u - P/2).^2 + (v - Q/2).^2);

H = 1 ./ (1 + (D_uv / D0).^2);% butterworth lowpass filter with n=1
filtered_spectrum = F .* H;
filtered_spectrum_shifted = ifftshift(filtered_spectrum);

filtered_image = abs(ifft2(filtered_spectrum_shifted));% Inverse DFT
filtered_image = filtered_image(1:M, 1:N);

corrected_im = smoothed - filtered_image;

%use laplacian of gaussian to segment edges
detectedCracks = edge(corrected_im, 'log');

originalImage = uint8(originalImage);


% Manually draw freehand region of interest (ROI)
figure;
imshow(originalImage);
title('Draw Freehand Region of Interest (ROI) containing cracks');

% Allow the user to draw a freehand region around the cracks
h = drawfreehand;
wait(h);
binaryMask = createMask(h);

% Convert binaryMask to the same data type as manualAnnotations
binaryMask = logical(binaryMask);

% Extract the selected region from the original image
roiImage = originalImage .* uint8(binaryMask);

detectedCracks = roiImage & detectedCracks;


% Visualize the original image, manual annotations, and detected cracks in the selected ROI
figure;
subplot(2, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(2, 2, 2);
imshow(manualAnnotations);
title('Manual Annotations');

subplot(2, 2, 4);
imshow(roiImage);
hold on;

% Overlay detected cracks on the selected ROI
[r, c] = find(detectedCracks .* binaryMask);
plot(c, r, 'r.', 'MarkerSize', 10); % Assuming red color for detected cracks
title('Detected Cracks in Freehand ROI');

hold off;

subplot(2, 2, 3);
imshow(detectedCracks)
title('Detected Cracks')

% Evaluate the performance using precision, recall, and F1 score on the selected ROI
truePositive = sum(sum(detectedCracks .* binaryMask & roiImage));
falsePositive = sum(sum(~detectedCracks .* binaryMask & roiImage));
falseNegative = sum(sum(detectedCracks .* binaryMask & ~roiImage));

precision = truePositive / (truePositive + falsePositive);
recall = truePositive / (truePositive + falseNegative);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
