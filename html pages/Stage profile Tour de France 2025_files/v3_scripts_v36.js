$(document).ready(function(){
  $( ".datepickerNormal" ).datepicker({ dateFormat: "yy-mm-dd" }); 
  $(document).on("click", '.show-all' ,function(e) {     e.preventDefault();    $(this).prev('table').find('tr').show();     $(this).hide();  });
  $(document).on("click", '.show-all-id a' ,function(e) {     e.preventDefault();    var contid = getUrlParameter('id',$(this).attr('href'));   $('.'+contid).find('li').show();  $(this).hide();  });
  $(document).on("click", '.clearPageFilter' ,function(e) {     e.preventDefault();   clearPageFilter();  });
  $('.h2hToggleColumn').click(function(e){    e.preventDefault();    $('.h2h').show();  });
  $('.h2hRider').click(function(e){  H2H($(this));  e.preventDefault();   });
  $('.rdrFilterSeason').click(function(e){    
    e.preventDefault();   $('.rdrSeasonNav li').removeClass('cur'); $(this).parents('li').addClass('cur');  
    $('#season').val($(this).data('season'));
    GetRiderResults($('#riderid').val(),$(this).data('season'),$('#sort').val(),$('#filter').val(),$('#discipline').val(),$('#pageid').val());
  });
  $('.rdrFilterSort').click(function(e){  e.preventDefault();  $('#sort').val($(this).data('sort'));    GetRiderResults($('#riderid').val(),$('#season').val(),$(this).data('sort'),$('#filter').val(),$('#discipline').val(),$('#pageid').val());  });
  $('.rdrFilterFilter').click(function(e){     e.preventDefault();       $('#filter').val($(this).data('filter'));   GetRiderResults($('#riderid').val(),$('#season').val(),$('#sort').val(),$(this).data('filter'),$('#discipline').val(),$('#pageid').val());  });
  $('.rdrFilterDiscipline').click(function(e){    e.preventDefault();    $('#discipline').val($(this).data('discipline'));   GetRiderResults($('#riderid').val(),$('#season').val(),$('#sort').val(),$('#filter').val(),$(this).data('discipline'),$('#pageid').val());  });

  $('.toggleSideMenu').click(function(e){    e.preventDefault();  $('.page-content').hide();  $('.side-nav').show(); $(this).hide(); });
  $('.backToContent').click(function(e){    e.preventDefault();  $('.page-content').show();  $('.side-nav').hide(); $('.toggleSideMenu').show(); });
  
  $('.filterResults').change(function(e){    e.preventDefault(); filterResults($(this).val(),$(this).data('type'));   });  
  $(document).on("click", '.selectResultTab' ,function(e) {     e.preventDefault();    selectResultTab($(this).data('id'),$(this).data('stagetype'),$(this).attr('href'));  });  
   $(document).on("click", '.selectResultTabAsync' ,function(e) {     e.preventDefault();     GetFullResults($('input[name="tab_id"]').val(),$('input[name="event_id"]').val(),$(this).data('id'),$(this).attr('href'));  });

  $('.viewGeneral').click(function(e){    e.preventDefault(); $('div.general').show(); $('div.today').hide();  });
  $('.viewToday').click(function(e){    e.preventDefault();  $('div.general').hide(); $('div.today').show(); });
  $('.resetResultFilter').click(function(e){    e.preventDefault();  resetResultFilter(); });
  $('.toggleResultsColumn').change(function(e){   if($(this).is(':checked')){ var val=1;   }else{  var val=0;   }   toggleResultsColumn($(this).data('code'),val);   });
  $(document).on("change",".toggleWonLost",function(e){      $('.timeWonLostComment').show();  if ($(this).is(':checked')) { $('.time_wonlost').show();  }else{  $('.time_wonlost').hide();  }   });
  $(".TimeWonLostFromRider").click(function(e){   e.preventDefault();   TimeWonLost($(this));  });
  $('.followRider').click(function(e){    e.preventDefault();  followRider($(this).find('a').attr('href')); });
  $('.snav a').click(function(e){    e.preventDefault(); $('.stab').hide();   var tab = getUrlParameter('snav',$(this).attr('href'));  $(this).parents('ul').find('div').removeClass('cur');  $(this).parent('div').addClass('cur'); $('.stab.'+tab+'').show();   });
  $(".showAllTeams").click(function(e){   e.preventDefault();       $('.rdr-teams2 li').show();     $('.rdr-teams2 li.combiLine').hide();       });
  $(".GetFullResults").click(function(e){   e.preventDefault();      GetFullResults($(this).data('tab_id'),$(this).data('event_id'),$('input[name="race_id"]').val(),$('input[name="race_seo"]').val());       });
  $(".showDiv").click(function(e){    e.preventDefault();    $("div[data-id="+$(this).attr('data-id')+"]").show();    $(this).hide();  });
  $(document).on("click",".viewMoreResults",function(e){          e.preventDefault();         $(this).parents('div.general').find('tbody tr:lt('+$(this).data('n')+')').show();    });
  $(document).on("change",".gotoH2H",function(e){  gotoH2H(); });
  
  /* autosearch */
  $("#search").autocomplete({     
    source: function(request, response) { 
      var data = QuickSearch($("#search").val());  
      $('#term').val($("#search").val());
      if(data['results'].length<1 && $("#search").val().length>2){     $.getJSON("resources/search.php", { term: $("#search").val(), quickresults: [] },     response);     }
      else if(data['results'].length<2 && $("#search").val().length>4){       $.getJSON("resources/search.php", { term: $("#search").val(), quickresults: [] },     response);     }
      else if(data['results'].length<4 && $("#search").val().length>6){      $.getJSON("resources/search.php", { term: $("#search").val(), quickresults: [] },     response);     }
      else{    response(data['results']);  }
      //console.log('N2: '+data['results'].length);
    },
    minLength: 1,
    autoFocus: true,
    delay: 100,
    select: function(event, ui){   
      if(ui.item.page == $('#search').data('page') && $('#search').data('url') != ''){    var url = $('#search').data('url').replace('-id-',ui.item.id);  }
      else if(ui.item.page != undefined){  var url = ui.item.page+'/'+ui.item.id;  } 
      else{  var url = ui.item.id;   }
      //console.log(ui.item.page+' = '+ui.item.id+' = '+$('#search').data('page')+' = '+url);
      window.location.href = url;   
    }     
  }); 

});

function TimeWonLost($this){
  var alpha = parseInt($this.data('s1')) - parseInt($this.data('s2'));
  $( "a.TimeWonLostFromRider" ).each(function( index ) {
    var beta = parseInt($(this).data('s1')) - parseInt($(this).data('s2'));
    var delta = alpha - beta;
    if(delta == 0){ var txt = 'st'; }
    else if(delta>0){  var txt = '<span class="green">+'+SecondToTime(delta,1)+'</span>';  }
    else if(delta<0){  var txt = '<span class="red">-'+SecondToTime(Math.abs(delta),1)+'</span>';  }
    $(this).html(txt);
  });
}

function gotoH2H(){
  var seo = seo1 = ''; 
  $('.results input:checked').each(function(i){
    $(this).parents('tr').find('a').each(function(k){
      var href = $(this).attr('href'); 
      if(href.substr(0,5) == 'rider'){   seo = href.substr(6);    }  
    });
    if(i==0){     seo1 = seo;     }
    else if(i==1){
      $('input:not(:checked)').hide(); 
      var html = ' <a href="rider-vs-rider/'+seo1+'/'+seo+'">GO</a>';
      $(this).parent('td').append(html);
    }
  });
}
function GetFullResults(tab_id,event_id,race,seo){
  //console.log('load async');
  $('.GetFullResults').remove(); 
  $('.tabnav li a').removeClass('selectResultTabAsync').addClass('selectResultTab');
  $.ajax({ method: "POST", url: "rce/results_v5.php", data: { getresults: 1,tab_id:tab_id, event_id: event_id }  }).done(function( response ) {   
    $('#resultsCont').html(response);
    selectResultTab(race,seo);   
  }) 
}
function followRider(url){
  var id = getUrlParameter('id',url); 
  var val = getUrlParameter('val',url);  
  var type = getUrlParameter('type',url); 
  $.ajax({ method:"GET",url:"resources/follow_entry.php",data:{id:id, type: type, val: val } }).done(function(response) {  console.log('doe iets');  $('.followRider').text(response);  }); 
}
function GetColumnIndex($this,column){
  var nr=-1;
  $this.find('thead th').each(function(k){      
    if($this.data('code') == column){ 
      nr = k+1;
    }
  }); 
  return nr;
}
function toggleResultsColumn(column,val){ 
  var columns = column.split(',');  
  for(var z=0; z<columns.length; z++){
    var column = columns[z];
    $('.'+column).show();
    $('.results').each(function(i){
      var nr=-1; 
      $(this).find('thead th').each(function(k){    
        if($(this).data('code') == column){ 
          nr = k+1;
        }
      });  
      if(val==1){
        $(this).find('th:nth-child('+nr+')').show();
        $(this).find('td:nth-child('+nr+')').show();
      }
      else{
        $(this).find('th:nth-child('+nr+')').hide();
        $(this).find('td:nth-child('+nr+')').hide();
      }
    });
  }
}

function resetResultFilter(){
  $('tr.filter').remove();
  $('tr').show();
   $('.filterResults').val('');
   $('.resetResultFilter').hide();
}
function selectResultTab(race,stagetype,seo){
  history.pushState("string", "page", seo);   
  $('.tabnav li').removeClass('cur');
  $('.tabnav li[data-id="'+race+'"]').addClass('cur');
  $('.resTab').hide();
  $('.resTab[data-id="'+race+'"]').show();
  $('.resTab[data-id="'+race+'"] div.general').show();
  $('.resTab[data-id="'+race+'"] div.today').hide();
 
  $('.resToggles').hide();
  $('.resToggles').each(function(i){
    if($(this).data('sts') != undefined){  
      if($(this).data('sts').indexOf(stagetype+',') > -1){ 
        $(this).show();
      } 
    }
  });
  if(stagetype==4 || stagetype==6 || stagetype==10){  
    $('.timeWonLostCont').show(); 
    TimeWonLost($(".TimeWonLostFromRider[data-st="+stagetype+"]").first()); 
  }
  if(stagetype==5 || stagetype==7){   
    $('.deltaPntCont').show();   
  }
}

function filterResults(value,type){   
  $('tr').show();
  $('tbody tr.filter').remove();
  $('.resetResultFilter').show();
  $('.results').each(function(i){
    var arr = [1,2,8,9];
    var arr2 = [1,8];

    var column = 'age'; 
    var nr=-1;    $(this).find('thead th').each(function(k){       if($(this).data('code') == column){       nr = k+1;     }    }); 

    var html = '';
    var prevtimelag = '';
    //var sec_winner = 0;
    $(this).find('tbody tr').each(function(k){     
       
     // if(k==0){ sec_winner =  parseInt($(this).find('td.time span.hide').text());   }   
      var go=0;
      if(value == 'wt'){      var str = 'bc="1"';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }
      else if(value == 'pro'){      var str = 'bc="2"';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }
      else if(value == 'ct'){      var str = 'bc="3"';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }
      else if(value == 'club'){      var str = 'bc="4"';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }
      else if(value == 'non_europe'){  var go=1;    var str = 'ct="EU"';        var find = $(this).html().indexOf(str);        if(find> -1){  go=0;  }         }
      else if(value == 'attackers'){    var str = 'svg_shield';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }
      else if(value == 'following'){    var str = 'followbadge';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }      
      else if(value == 'gc_favorites'){    var str = 'fav_gc';        var find = $(this).html().indexOf(str);        if(find> -1){  go=1;  }         }      
      else if(type=='custom'){
        if(value == 'age_u23'){  var age = parseInt($(this).find('td:nth-child('+nr+')').text());  if(age<23){ go=1;  }}
        if(value == 'age_u25'){  var age = parseInt($(this).find('td:nth-child('+nr+')').text());  if(age<25){ go=1;  }}
        if(value == 'age_o30'){  var age = parseInt($(this).find('td:nth-child('+nr+')').text());  if(age>=30){ go=1;  }}
        if(value == 'age_o35'){  var age = parseInt($(this).find('td:nth-child('+nr+')').text());  if(age>=35){ go=1;  }}
      }
      else{
        if(type=='nation'){  var str = 'flag '+value;  }else{  var str = value; }
        var find = $(this).html().indexOf(str);
        if(find> -1){  go=1;  }
      } 
      if(go==1){   
        //var db_sec = parseInt() - sec_winner; 
        //if(prevsec != db_sec){  var time = SecondToTime(db_sec,1);  }else{  var time = ',,'; }
        //prevsec = db_sec;
        var timelag = $(this).find('td.time span.hide').text();
        if(timelag == prevtimelag){  timelag_title = ',,'; }else{  timelag_title = timelag; }
        
        $(this).find('td.time font').text(timelag_title);
        prevtimelag = timelag;

        var tr = $(this).html();   

        if(jQuery.inArray(k,arr) !== -1 && 1==2){
          if(jQuery.inArray(k,arr2) !== -1){  var v = $(this).next().find('td:first-child div').text();  }else{ var v = $(this).prev().find('td:first-child div').text();  }
          var tr2 = tr.substring(tr.indexOf('</td>')); 
          html = html + '<tr class="filter"><td><div>'+v+'</div>'+tr2+'</tr>';
        }
        else{  html = html + '<tr class="filter">'+tr+'</tr>';}        
      } 
    }); 
    $(this).find('tbody').append(html);
    $(this).find('tbody tr.filter').show();
    $(this).find('tbody tr').not('.filter').hide();
  });
}

function clearPageFilter(){ 
  $('.filter li').each(function(i){
    var val = $(this).attr('data-def');  
    if(val == 'curyear'){  val = new Date().getFullYear(); }
    else if(val == 'today'){  var d = new Date(); var val = d.getFullYear() + "-" + (d.getMonth()+1) + "-" + d.getDate();    } 
    $(this).find('span.inputCont select').val(val);
    $(this).find('span.inputCont input[type="text"]').val(val);
  });
}

function H2H($this){  
    var seo = $this.data('seo');  
    var $td = $this.parent('td');
    $('.h2hRider').each(function(i){
      $(this).next().attr('href','rider-vs-rider/'+seo+'/'+$(this).data('seo'));
    });
    $('.h2hRider').hide();
    $('.h2hGoto').show(); 
    $td.find('.h2hGoto').hide();
}

function GetRiderResults(id,season,sort,filter,discipline,pageid){
  var url = 'rider/'+$('#seo').val()+'/'+season;
  history.pushState("string", "page", url);   
  $('.rdrResults tbody').html('');
  $('.rdrResults tbody').css('height','200px');
  $('.rdrSeasonSum').hide();
  //console.log(season+' = '+sort+' = '+filter+' = '+discipline);
  $('#rdrResultCont').html('loading results');
  $.ajax({ method: "POST", url: "https://www.procyclingstats.com/rdr/start4.php", data: { getresults: 1,id:id, season: season, sort: sort, xfilter: filter, dis:discipline, pageid: pageid }  }).done(function( response ) {   $('#rdrResultCont').html(response);   }) 
}

function QuickSearch(term){  
  term = term.toLowerCase();
  var n = 0; 
  var arr = [];
  var slugterm = slug(term);
  var parts = slugterm.split('-'); //  

  for(var i=0; i<search.length; i++){
    var e = search[i];

    var str = slug(e[2]);  //  
    var go = nMatches = 0; 

    for(z=0; z<parts.length; z++){     if(str.toLowerCase().indexOf(parts[z]) > -1){   nMatches++;  } }  //  
    if(nMatches==parts.length){  go = 1; }
   
    if(go==1){ // quick check
      n++;
      if(n<200){  
        var names = search[i][2].split(',');
        var nameformats = [names[0]+' '+names[1]+' '+names[2], names[2]+' '+names[0]+' '+names[1], names[1]+' '+names[2]];

        var score = 0; 
        for(var k=0; k<nameformats.length; k++){
          var str = slug(nameformats[k]);  
          if(str.substr(0,slugterm.length) == slugterm){  score = 100-k;   break;  }
        }     
        arr.push({'page':e[0],'label':e[2].replace(',',''),'id':e[1],'score':score, 'fromcache':1,'eid':e[3]});
      }
      else{  break;  }
    }
  }
   
  arr.sort(keysrt('score'));
  arr = arr.slice(0, 10);   
 
  var res = [];
  res['highestScore'] = 100;
  res['results'] = arr; 
  return res;
}

var slug = function(str,char='-') {
  str = str.replace(/^\s+|\s+$/g, ''); // trim
  str = str.toLowerCase();

  // remove accents, swap ñ for n, etc
  var from = "ãàáäâẽèéëêìíïîõòóöôùúüûñç·/_,:;žØøČčŠšý";
  var to   = "aaaaaeeeeeiiiiooooouuuunc------zooccssy";
  for (var i=0, l=from.length ; i<l ; i++) {
    str = str.replace(new RegExp(from.charAt(i), 'g'), to.charAt(i));
  }

  str = str.replace(/[^a-z0-9 -]/g, '') // remove invalid chars
    .replace(/\s+/g, char) // collapse whitespace and replace by -
    .replace(/-+/g, char); // collapse dashes

  return str;
};

function keysrt(key,ascdesc='desc') {
  return function(a,b){
    if(ascdesc == 'desc'){
      if (a[key] < b[key]) return 1;
      if (a[key] > b[key]) return -1;
      return 0;
    }
    else {
      if (a[key] > b[key]) return 1;
      if (a[key] < b[key]) return -1;
      return 0;
    }
  }
}
function getUrlParameter(sParam, sUrl) {
  var sPageURL = sUrl.replace('?','&');
  var sURLVariables = sPageURL.split('&');
  for (var i = 0; i < sURLVariables.length; i++) 
  {
      var sParameterName = sURLVariables[i].split('=');
      if (sParameterName[0] == sParam) 
      {
          return sParameterName[1];
      }
  }
} 

function getParams(url) {
  var queryString = url.substring(url.indexOf('?') + 1);
  var paramsArr = queryString.split('&'); 
  var keyvalues = [];

  for (var i = 0, len = paramsArr.length; i < len; i++) {
      var keyValuePair = paramsArr[i].split('='); 
      keyvalues[keyValuePair[0]] = keyValuePair[1];
  }

  return keyvalues;
}
function RVC(id){
  $.ajax({ method: "GET", url: "https://www.procyclingstatsdata.com/views.php", data: { v: 'rider', id: riderid, uid: '3' }   }).done(function(data) {  console.log(data);    });
}

function SecondToTime(s,stripzeros=0){
  var h = Math.floor(s / 3600); //Get whole hours
  s -= h * 3600;
  var m = Math.floor(s / 60); //Get remaining minutes
  s -= m * 60;
  s = Math.round(s);
  var result = h + ":" + (m < 10 ? '0' + m : m) + ":" + (s < 10 ? '0' + s : s); //zero padding 
  if(stripzeros == 1){  
    var result = ' '+result;
    result = result.replace(' 0:0','');
    result = result.replace(' 0:','');
  } 
  return result;
}

function makeid(lengte) {
  var text = "";
  var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (var i = 0; i < lengte; i++)
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  return text;
}

function ListSearch(term,list,type,maxlength=10) {
  var res = [];  
  var termWords = term.split(/\s+/);
 
  for(var i=0; i<list.length; i++){
    if(type==list[i]['type'] || type=='all'){ // zoeken in opgegeven type

      if(list[i]['keywords'] != undefined){ var keywords = list[i]['keywords'].toLowerCase();  }
      else if(list[i]['label'] == undefined){ var keywords = ''; }
      else{   var keywords = list[i]['label'].toLowerCase();   }
      keywords = SwapSpecialCharsV1(keywords);

      if (keywords.indexOf(term) >= 0){       res.push(list[i]);       } 
      else{
        var n=0; 
        for(var k=0; k<termWords.length; k++){   if (keywords.indexOf(termWords[k]) >= 0){  n++;  }       }  // zoek op afzonderlijke termen
        if(n==termWords.length){             res.push(list[i]);        } // als elke term in de zin zit, toevoegen aan resultaten
      }
      if(res.length>=maxlength){  break; }  // als aantal resultaten gelijk is aan maximum, stop met verzamelen
    }
  } 
  return res;
} 

function SwapSpecialCharsV1(str) {
  // remove accents, swap ñ for n, etc
  var from = "ãàáäâěẽèéëêìíïîīõòóöôùúüûūñņçć·/_,:;žØøČčŠšýŁłŃĻļ'";   
  var to   = "aaaaaeeeeeeiiiiiooooouuuuunncc------zooccssyllnll ";
  for (var i=0, l=from.length ; i<l ; i++) {
    str = str.replace(new RegExp(from.charAt(i), 'g'), to.charAt(i));
  }
  return str;
}

function TimeToSeconds(time){
  var a = time.split(':'); // split it at the colons 
  if(a.length==2){  var seconds = a[0]*60 + parseInt(a[1]);     }
  else{  var seconds = (+a[0]) * 60 * 60 + (+a[1]) * 60 + (+a[2]);  }
  return seconds;
}

function SecToTimeArray(total_sec){
  var days = days0 = Math.floor(total_sec/86400);
  if(days<10){ var days0 = '0'+days;  }
  var hours_sec = (total_sec - days*86400);
  var hours = hours0 = Math.floor(hours_sec/3600);
  if(hours<10){ var hours0 = '0'+hours;  }
  var minute_sec = hours_sec - hours*3600; 
  var minutes = minutes0 = Math.floor(minute_sec / 60);
  if(minutes<10){ var minutes0 = '0'+minutes;  }
  var seconds = seconds0 = minute_sec - minutes*60;
  if(seconds<10){ var seconds0 = "0"+seconds+"";  }
  var str = days0+hours0+minutes0+seconds0;
  var res = [""+days0+"",""+hours0+"",""+minutes0+"",""+seconds0+"",""+str+""];
  return res;
}

function MilliSecondsTime(time,format=1){
  var time = Math.abs(time);
  var ms = time % 1000; 
  var ms = ""+ms; 
  if(ms.length==2){  var ms = '0'+ms; }
  else if(ms.length==1){  var ms = '00'+ms; } 
  var ms = ""+ms+"";   
  if(ms.substr(2,1)=='0'){  var ms = ms.substr(0,2);  }
   
  var t = SecondToTime(Math.floor(time/1000),1); 
  t = t.replace(':','.');  
  if(format == 1){  if(ms>0){  var res = t+','+ms;  }else{  var res = t;  }  }
  else if(format == 2){   if(ms>0){  var res = t+',<font style="font-size: 10px;  ">'+ms+'</font>';  }else{  var res = t;  }    } 
  return res;  
}