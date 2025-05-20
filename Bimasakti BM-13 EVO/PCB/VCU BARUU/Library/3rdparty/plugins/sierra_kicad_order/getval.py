import os
import urllib.request, urllib.error, urllib.parse
import json
import logging
from . import  globalVars
import configparser
import pcbnew



def getValidate(boardName,vuserid,vtoken,sessionid,userName):


    try:
        infile2 = open(boardName, 'rb')
        ledata2 = {'file': infile2}
        file_name =  boardName.replace('\\','/')
        file_name = file_name.rpartition('/')[2]
        headers = {
                'Authorization': vtoken ,
                'user_id' : vuserid,
                'User-email' : userName,
                'file_name': file_name,
                'Content-Type':'application/zip',
                'Content-Length':os.stat(boardName).st_size,
                'bomAutoMap' : 'false',
                'sessionId' : sessionid,
                'source' : globalVars.pluginVersion
                    }
        request = urllib.request.Request(globalVars.kicadBackendURL  + "/validate?service=validateFile",  data=open(boardName, 'rb'),headers=headers)


        response = urllib.request.urlopen(request)
        return(json.loads(response.read()))
    except Exception as e:
        return False

def getorder(userID):

    res = urllib.request.Request(globalVars.SCABackendURL + "/orderapi/order/getListOfOrderDetails/userID={}".format(userID))
    response = urllib.request.urlopen(res)
    return(json.loads(response.read()))

def redirectCustPortal():
    redirectCustPortalrl = globalVars.FEMSBackendURL + "/customer-portal/"
    return(redirectCustPortalrl)

def logout():

    config_path = pcbnew.SETTINGS_MANAGER.GetUserSettingsPath()
    config_file = os.path.join(config_path, "sierra_plugin.cfg")
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        
        with open(config_file, 'w') as configfile:
            config.write(configfile)
        return True
    except:
        return False
    

def redirectSierraPortal():
    redirectSierraPortal = globalVars.SierraSite
    return(redirectSierraPortal)


def redirectkicadQuotePlugin():
    redirectkicadQuotePlugin = globalVars.SierraSite + 'kicad-quote-plugin/'
    return(redirectkicadQuotePlugin)
