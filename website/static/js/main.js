$(document).ready(function () {
    $('#result').hide();
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#username-form')[0]);
        // Show loading animation
        $('.loader').show();
        $('#result').hide();
        $('.image-result-container').remove();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                const dataJson = JSON.parse(data);
                console.log(dataJson);
                const { username, fullname, avatar, photos, result } = dataJson;
                $('.prediction-result').text(result);
                $('.username-value').text(username.value);
                $('.username-m').text(parseInt(username.man * 100));
                $('.username-f').text(parseInt(username.woman * 100));
                $('.fullname-value').text(fullname.value);
                $('.fullname-m').text(parseInt(fullname.man * 100));
                $('.fullname-f').text(parseInt(fullname.woman * 100));
                $('.avatar').attr('src', avatar.value);
                $('.avatar-m').text(parseInt(avatar.man * 100));
                $('.avatar-f').text(parseInt(avatar.woman * 100));
                const imageContainer = $('.image-result');
                for(const photo of photos) {
                    const photoUrl = photo.value;
                    const m = parseInt(photo.man * 100);
                    const w = parseInt(photo.woman * 100);
                    const photoElement = $('<div class="image-result-container">\n' +
                        '                <img src="'+ photoUrl +'" class="photo-prediction-result" />\n' +
                        '                <p>M: <b class="photo-m">'+ m +'</b></p>\n' +
                        '                <p>F: <b class="photo-f">'+w+'</b></p>\n' +
                        '            </div>');
                    imageContainer.append(photoElement)
                }
                $('#result').fadeIn(600);

            },
        });
    });

});
