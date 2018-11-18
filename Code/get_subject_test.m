function testing_data = get_subject_test(filename,total_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = 1:total_size
        imageI = faces(:,:,3*n-2);
        data = [data imageI(:)];
    end
    testing_data = data;
end