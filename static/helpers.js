function analyze(){
    img_data = document.getElementById('my_result').src;
    var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
    // console.log(img_data);
    $.ajax({
        data: img_data,
        type: 'POST',
        url : $SCRIPT_ROOT + '/predict/',

        success: function(response){
            console.log(typeof response) 
              $('result_text').text(response) 
          },
    })
}

