testFeaturesDir = dir("test_features/*.mat");
SortedTestImage = natsortfiles({testFeaturesDir.name});
classifiers = load("classifiers.mat");
testLabels = load('testLabels.mat');
classifiers = classifiers.classifiers;
boundingBoxes = load("boundingBoxes.mat");

boundingBoxes = boundingBoxes.bb;
classes = zeros(size(testFeaturesDir, 1), 8);
testLabels = testLabels.testLabels;
ConfusionM = zeros(10,10);
groundTruthBoxes = importdata("test/bounding_box.txt");
for i = 1:size(testFeaturesDir, 1)
    featureArray = load(strcat(strcat("test_features/", int2str(i-1)), ".mat"));    
    oldfeature = featureArray.feature;
    feature(1:size(oldfeature, 1),1:size(oldfeature, 3)) = oldfeature(1:size(oldfeature, 1),1,1:size(oldfeature, 3));
    clear oldfeature
    maxScore = -234234;
    class = -1;
    selectedWindowIndex = 0;
    c = -1;
    
    for j = 1:size(classifiers, 2)
        cl = classifiers(1, j);
        cl = cl{1, 1};
        [assignments, scores] = predict(cl, feature);
        
        [ maxValue, maxIndex] = max(scores(:, 2));
        
        if maxValue > maxScore
            maxScore = maxValue;
            selectedWindowIndex = maxIndex;
            class = j;
        end
        
    end
    classes(i, 1) = class;
    classes(i, 2) = maxScore;
    classes(i, 3) = selectedWindowIndex;
    x = boundingBoxes(1, i);
    x = x{1, 1};
    classes(i, 4:8) = x(selectedWindowIndex, :);
   
   
   
    truthclass = groundTruthBoxes.textdata(i);
    if truthclass=="n01615121"
        truthclassNo = 1;
    elseif truthclass =="n02099601"
        truthclassNo = 2;
    elseif truthclass =="n02123159"
        truthclassNo = 3;
    elseif truthclass =="n02129604"
         truthclassNo = 4;
    elseif truthclass =="n02317335"
         truthclassNo = 5;
    elseif truthclass =="n02391049"
         truthclassNo = 6;
    elseif truthclass =="n02410509"
         truthclassNo = 7;
    elseif truthclass =="n02422699"
         truthclassNo = 8;
    elseif truthclass =="n02481823"
        truthclassNo = 9;
    elseif truthclass =="n02504458"
        truthclassNo = 10;
    end 
    ConfusionM(class,truthclassNo) = ConfusionM(class,truthclassNo)+1;
end

g = groundTruthBoxes.data;
b = classes(:,4:7);
overlapRatio = zeros(100,1);
localizationResults = zeros(100,1);

for i = 1: size(g)
   intersectionArea = rectint(g(i,:),b(i,:));
   unionArea = g(i,3)*g(i,4) + b(i,3)*b(i,4) - intersectionArea;
   overlapRatio(i) = intersectionArea / unionArea;

   if overlapRatio(i) >= 0.5
      localizationResults(i) = 1;
   else
      localizationResults(i) = 0;
   end
end

accuracy = 0;
for i = 1: size(g)
    if localizationResults(i) == 1;
        accuracy = accuracy + 1;
    end
end

accuracy = accuracy / size(g,1) * 100;
