var api_root_url = location.hostname=='localhost' ?  'http://localhost:5000' : 'https://dopelemon.me/api'
var res_folder_root_url = location.hostname=='localhost' ? '' : 'https://dopelemon.me/'


function invoke_upload_image(){
    $('#upload-btn').click();
}

function append_resultant_image_to_dom(res_url){
    var res_dom_container_id = 'result'
    $('#'+res_dom_container_id).empty()
    $('#'+res_dom_container_id).append('<img src='+res_url+' />')
}

function upload_image(){
    var res_dom_container_id = 'result'
    var input_elem = document.querySelector('input[type=file]')
    var file = input_elem.files[0]; //sames as here
    var form_data = new FormData();
    form_data.append('image', file)
    
    var request = new XMLHttpRequest();
    request.open('POST', api_root_url+'/predictMultiClass');
    request.send(form_data);
    
    $('#'+res_dom_container_id).empty()
    $('#'+res_dom_container_id).append('<div style=\'width:100%;text-align:center;padding-top:150px;\'><img style=\'width:50px;\' src=\'images/spinner.gif\'/></div>')
    input_elem.value = ''
    
    request.onreadystatechange = function(e) {
        if(this.readyState==4 && this.status==200){
            var response = JSON.parse(request.responseText)
            if(response){
                append_resultant_image_to_dom(res_folder_root_url+response['res_url'])
            }
        }
        else if(this.readyState==4){
            $.toaster({ settings: {timeout:5000}, title:'', message : 'An error occurent while processing the image', priority : 'danger' });   
            $('#'+res_dom_container_id).empty()
            $('#'+res_dom_container_id).append('<img src=\'images/multi-class/multi-class-sample.png\'/>')
        }
      };
}
