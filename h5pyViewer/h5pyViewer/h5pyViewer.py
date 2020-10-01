#!/usr/bin/env python
#*-----------------------------------------------------------------------*
#|                                                                       |
#|  Copyright (c) 2013 by Paul Scherrer Institute (http://www.psi.ch)    |
#|                                                                       |
#|              Author Thierry Zamofing (thierry.zamofing@psi.ch)        |
#*-----------------------------------------------------------------------*
'''
hdf5 viewer to dispay images, tables, attributes and trees of a hdf5 file.
'''
import os,sys
import wx,h5py
import wx.py
from hdfTree import *
from hdfGrid import *
from hdfAttrib import *
from hdfImage import  *
try:
  from hdfImageGL import  *
except ImportError as e:
  print 'ImportError: '+e.message
try:
  from FrmPyFAI import  *
except ImportError as e:
  print 'ImportError: '+e.message
try:
  from FrmProcRoiStat import ProcRoiStatFrame
except ImportError as e:
  print 'ImportError: '+e.message

import utilities as ut


class AboutFrame(wx.Frame):
  def __init__(self,parent):
    wx.Frame.__init__(self,parent,-1,'About h5pyViewer',size=(300,330))
    imgDir=ut.Path.GetImage()
    icon = wx.Icon(os.path.join(imgDir,'h5pyViewer.ico'), wx.BITMAP_TYPE_ICO)
    self.SetIcon(icon)
    self.Centre()
    panel=wx.Panel(self,-1)
    import pkg_resources

    v=pkg_resources.get_distribution("h5pyViewer")
    s='Version:'+str(v)+'\n(c) www.psi.ch\n Author: Thierry Zamofing\n thierry.zamofing@psi.ch'

    st0=wx.StaticText(panel,-1,s,(30,10))
    bmp = wx.StaticBitmap(panel,-1,wx.Bitmap(os.path.join(imgDir,'splash1.png'), wx.BITMAP_TYPE_ANY ), (30,st0.Position[1]+st0.Size[1]+10))

    for k,v in os.environ.iteritems():
      print k,'=',v

class HdfTreePopupMenu(wx.Menu):
  def __init__(self, wxObjSrc):
    wx.Menu.__init__(self)
    self.wxObjSrc=wxObjSrc
    self.AddMenu(self.OnShowAttrib,"Show Attributes")
    self.AddMenu(self.OnShowData,"Show Data")
    self.AddMenu(self.OnShowImage,"Show Image")
    self.AddMenu(self.OnShowImageGL,"Show Image OpenGL")
    self.AddMenu(self.OnShowImgFAI1D,"Show Azimutal Integral Image 1D")
    self.AddMenu(self.OnShowImgFAI2D,"Show Azimutal Integral Image 2D")
    self.AddMenu(self.OnShowRoiStat,"Show Roi Statistics")
    self.AddMenu(self.OnShell,"Python Shell")
    self.AddMenu(self.OnPrintProperties,"Print Properties")
    self.AddMenu(self.OnItem2,"Item Two")
    self.AddMenu(self.OnItem3,"Item Three")

  def AddMenu(self,func,lbl):
    item = wx.MenuItem(self, -1, lbl)
    self.AppendItem(item);
    self.Bind(wx.EVT_MENU, func, item)
    return item

  def OnShowAttrib(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    if type(hid)==h5py.h5f.FileID:
      hid=h5py.h5o.open(hid,'/')
    frame=HdfAttribFrame(wxTree,lbl,hid)
    frame.Show(True)

  def OnShowData(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    frame=HdfGridFrame(wxTree,lbl,hid)
    frame.Show(True)

  def OnShowImage(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    frame=HdfImageFrame(wxTree,lbl,hid)
    frame.Show(True)

  def OnShowImageGL(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    frame=HdfImageGLFrame(wxTree,lbl,hid)
    frame.Show(True)

  def OnShowImgFAI1D(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    frame=HdfPyFAI1DFrame(wxTree,lbl,hid)
    frame.Show(True)

  def OnShowImgFAI2D(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    frame=HdfPyFAIFrame(wxTree,lbl,hid)
    frame.Show(True)

  def OnShowRoiStat(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    if type(hid)==tuple: hid=hid[0] #external link->get dataset
    dlg = wx.FileDialog(wxTree, "Choose valid mask file (e.g. pilatus_valid_mask.mat)", os.getcwd(), '','MATLAB files (*.mat)|*.mat|all (*.*)|*.*', wx.FD_OPEN|wx.FD_CHANGE_DIR)
    if dlg.ShowModal() == wx.ID_OK:
      fnValMsk= dlg.GetPath()
      print 'OnOpen',fnValMsk
    dlg.Destroy()
    if not fnValMsk: return
    dlg = wx.FileDialog(wxTree, "Choose ROI mask file (e.g. pilatus_integration_mask.mat)", os.getcwd(), '','MATLAB files (*.mat)|*.mat|all (*.*)|*.*', wx.FD_OPEN|wx.FD_CHANGE_DIR)
    if dlg.ShowModal() == wx.ID_OK:
      fnIntegMsk = dlg.GetPath()
      print 'OnOpen',fnIntegMsk
    dlg.Destroy()
    if not fnIntegMsk: return
    #fnMatRoi='/scratch/detectorData/cSAXS_2013_10_e14608_georgiadis_3D_for_Marianne/analysis/data/pilatus_integration_mask.mat'
    frame=ProcRoiStatFrame(wxTree,lbl,hid,fnValMsk,fnIntegMsk)
    frame.Show(True)

  def OnShell(self, event):
    wxTree,wxNode=self.wxObjSrc
    frame = wx.Frame(wxTree, -1, "wxPyShell",size=wx.Size(800, 500))
    imgDir=ut.Path.GetImage()
    icon = wx.Icon(os.path.join(imgDir,'h5pyViewer.ico'), wx.BITMAP_TYPE_ICO)
    frame.SetIcon(icon)
    frame.Centre()
    wnd=app.GetTopWindow()
    loc={'app'  :app,
         'fid'  :app.GetTopWindow().fid,
         'lbl'  :wxTree.GetItemText(wxNode),
         'hid'  :wxTree.GetPyData(wxNode),
         'h5py' : h5py
         }
    introText='''Shell to the HDF5 objects
app: application object
fid: hdf5 file object
lbl: label of selected hdf5 object
hid: selected hdf5 object

#Examples:
import h5py
ds=h5py.Dataset(hid)
ds[1,:,:]

#using user defined modules
import userSample as us;reload(us);us.test1(hid)

'''
    shell=wx.py.shell.Shell(frame, introText=introText,locals=loc)
    frame.Show(True)
    #if loc is None, all variables are visible. the context is global
    #shell.push('wnd=app.GetTopWindow()')
    #for cmd in [
    #  'wnd=app.GetTopWindow();wxTree=wnd.wxTree',
    #  'wxNode=wnd.wxTree.GetSelection()',
    #  'print wnd.fid',
    #  'lbl=wxTree.GetItemText(wxNode)',
    #  'hid=wxTree.GetPyData(wxNode)']:
    #  shell.run(cmd, prompt=False)

  def OnPrintProperties(self, event):
    wxTree,wxNode=self.wxObjSrc
    lbl=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    print HdfViewerFrame.GetPropertyStr(wxTree,wxNode)

  def OnItem2(self, event):
    print 'OnItem2'
    pass

  def OnItem3(self, event):
    print 'OnItem3'
    pass

class HdfViewerFrame(wx.Frame):

  def OpenFile(self,fnHDF):
    try:
      self.fid=h5py.h5f.open(fnHDF,flags=h5py.h5f.ACC_RDONLY)
    except IOError as e:
      sys.stderr.write('Unable to open File: '+fnHDF+'\n')
    else:
      self.wxTree.ShowHirarchy(self.fid)

  def CloseFile(self):
    #http://docs.wxwidgets.org/2.8/wx_windowdeletionoverview.html#windowdeletionoverview
    #print 'CloseFile'
    try:
      self.fid.close()
      del self.fid
    except AttributeError as e:
      pass

  def __init__(self, parent, title):
    wx.Frame.__init__(self, parent, title=title, size=wx.Size(650, 350))
    imgDir=ut.Path.GetImage()
    icon = wx.Icon(os.path.join(imgDir,'h5pyViewer.ico'), wx.BITMAP_TYPE_ICO)
    self.SetIcon(icon)
    wxSplt = wx.SplitterWindow(self, -1)
    wxTree = HdfTreeCtrl(wxSplt, 1, wx.DefaultPosition, (-1,-1),  wx.TR_HAS_BUTTONS)
    wxTree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelChanged, id=1)
    wxTree.Bind(wx.EVT_TREE_ITEM_MENU, self.OnMenu, id=1)
    #wx.EVT_TREE_ITEM_MENU(id, func)
    wxTxt = wx.StaticText(wxSplt, -1, '',(10,10) )#, style=wx.ALIGN_CENTRE)

    wxSplt.SplitVertically(wxTree, wxTxt)
    wxSplt.SetMinimumPaneSize(320)
    #wxLstCtrl=HdfAttrListCtrl(wxSplt)
    #wxSplt.SplitVertically(wxTree, wxLstCtrl)
    self.BuildMenu()

    self.Centre()

    self.wxTree=wxTree
    self.display=wxTxt
  def __del__(self):
    self.CloseFile()

  def OnOpen(self, event):
    dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), '','HDF5 files (*.hdf5;*.h5)|*.hdf5;*.h5|all (*.*)|*.*', wx.FD_OPEN|wx.FD_CHANGE_DIR)
    if dlg.ShowModal() == wx.ID_OK:
      path = dlg.GetPath()
      #mypath = os.path.basename(path)
      #self.SetStatusText("You selected: %s" % mypath)
      self.CloseFile()
      self.OpenFile(path)
      #print 'OnOpen',path
    dlg.Destroy()

  def OnCloseWindow(self, event):
    #print 'OnCloseWindow'
    self.Destroy()

  def OnAbout(self,event):
    frame=AboutFrame(self)
    frame.Show()

  def BuildMenu(self):
    #http://wiki.wxpython.org/AnotherTutorial#wx.MenuBar
    mnBar = wx.MenuBar()

    #-------- File Menu --------
    mn = wx.Menu()
    mnItem=mn.Append(wx.ID_OPEN, '&Open', 'Open a new document');self.Bind(wx.EVT_MENU, self.OnOpen, mnItem)
    #mnSub = wx.Menu()
    #mnItem=mnSub.Append(wx.ID_ANY, 'SubMenuEntry', 'My SubMenuEntry')
    #mn.AppendMenu(wx.ID_ANY, 'SubMenu', mnSub)
    mn.AppendSeparator()
    mnItem=mn.Append(wx.ID_EXIT, '&Quit', 'Quit the Application');self.Bind(wx.EVT_MENU, self.OnCloseWindow, mnItem)
    mnBar.Append(mn, '&File')

    self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

    #-------- Edit Menu --------
    #mn = wx.Menu()
    #mnBar.Append(mn, '&Edit')

    #-------- Help Menu --------
    mn = wx.Menu()
    #mnItem=mn.Append(wx.ID_HELP,'Help','Application Help')
    mnItem=mn.Append(wx.ID_ABOUT,'About','Application About');self.Bind(wx.EVT_MENU, self.OnAbout, mnItem)
    mnBar.Append(mn, '&Help')

    #mn.AppendSeparator()
    #mnItem = wx.MenuItem(mn, 105, '&Quit\tCtrl+Q', 'Quit the Application')
    #mnItem.SetBitmap(wx.Image('stock_exit-16.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap())
    #mn.AppendItem(mnItem)
    self.SetMenuBar(mnBar)
    self.CreateStatusBar()

  @staticmethod
  def GetPath(wxTree,wxNode):
    if wxTree.GetRootItem()==wxNode:
      hid=wxTree.GetPyData(wxNode)
      return hid.name
    wxNodeParent=wxTree.GetItemParent(wxNode)
    if wxTree.GetRootItem()==wxNodeParent:
      return wxTree.GetItemText(wxNode)
    else:
      return HdfViewerFrame.GetPath(wxTree,wxNodeParent)+'/'+wxTree.GetItemText(wxNode)

  @staticmethod
  def GetPropertyStr(wxTree,wxNode):

    path=str(HdfViewerFrame.GetPath(wxTree,wxNode))

    hidStr=wxTree.GetItemText(wxNode)
    hid=wxTree.GetPyData(wxNode)
    #o=wxTree.GetItemData(wxNode)
    #print o.Data,wxTree.GetPyData(wxNode)
    #if type(gid)==h5py.h5g.GroupID:
    txt=path+'\n'
    t=type(hid)
    if t==tuple:
      if hid[0]==None:
        txt+='missing external link:\n  '+hid[1]+'\n  '+hid[2]
        return txt
      else:
        txt+='external link:\n  '+hid[1]+'\n  '+hid[2]
        hid=hid[0]
        t=type(hid)
    if t==h5py.h5f.FileID:
      txt+=type(hid).__name__+':%d\n'%hid.id
      hid=h5py.h5o.open(hid,'/')
      t=type(hid)
    objInf=h5py.h5o.get_info(hid)
    #print t,hid.id,objInf.fileno, objInf.rc, objInf.type, objInf.addr, objInf.hdr
    txt+=type(hid).__name__+':%d\n'%hid.id
    txt+='addr:%d fileno:%d refCnt:%d\n'%(objInf.addr,objInf.fileno, objInf.rc)
    try:
      wxNodeParent=wxTree.GetItemParent(wxNode)
      txtParent=wxTree.GetItemText(wxNode)
      dataParent=wxTree.GetPyData(wxNode)
      gid=wxTree.GetPyData(wxNodeParent)
      softLnk=gid.get_linkval(hidStr)
    except BaseException as e:
      pass
    else:
      txt+='Soft Link:'+softLnk+'\n'
    try: numAttr=h5py.h5a.get_num_attrs(hid)
    except ValueError as e:
      pass
    else:
      if numAttr>20:
        txt+='Attributes:%d (too many to show)\n'%numAttr
      else:
        txt+='Attributes:%d\n'%numAttr
        for idxAttr in range(numAttr):
          aid=h5py.h5a.open(hid,index=idxAttr)
          txt+='\t'+aid.name+'\t'+str(GetAttrVal(aid))+'\n'
    val=None
    if t==h5py.h5g.GroupID:
      pass
    elif t==h5py.h5d.DatasetID:
      txt+='\nshape: '+str(hid.shape)+'\n'
      tt=hid.get_type()
      ttt=type(tt)
      if ttt==h5py.h5t.TypeCompoundID:
        txt+='type: Compound\n'
      elif ttt==h5py.h5t.TypeStringID:
        sz=tt.get_size()
        txt+='type: String (length %d)\n'%sz
      else:
        txt+='type: '+str(tt.dtype)+'\n'


      pl=hid.get_create_plist()
      txFcn=(
       ('chunk',h5py.h5p.PropDCID.get_chunk),
       ('fill time',   h5py.h5p.PropDCID.get_fill_time),
       ('alloc_time',  h5py.h5p.PropDCID.get_alloc_time),
       #('class',       h5py.h5p.PropDCID.get_class),
       #('fill_value',  h5py.h5p.PropDCID.get_fill_value),
       #('filter',      h5py.h5p.PropDCID.get_filter),
       #('filter_by_id',h5py.h5p.PropDCID.get_filter_by_id),
       ('layout',      h5py.h5p.PropDCID.get_layout),
       ('nfilters',    h5py.h5p.PropDCID.get_nfilters),
       #('obj_track_times', h5py.h5p.PropDCID.get_obj_track_times),
       )
      for tx,func in txFcn:
        try: v=func(pl)
        except ValueError as e: pass
        else:txt+=tx+':'+str(v)+'\n'

      if hid.shape==() or np.prod(hid.shape)<10: #show up to max. 10 element arrays
      #if  ttt==h5py.h5t.TypeStringID or hid.shape==() or hid.shape==(1,):
        ds=h5py.Dataset(hid)
        txt+='Value:\n\t'+str(ds.value)+'\n'

    return txt

  def OnSelChanged(self, event):
      wxNode =  event.GetItem()
      txt=HdfViewerFrame.GetPropertyStr(self.wxTree,wxNode)
      self.display.SetLabel(txt)

  def OnMenu(self, event):
      wxNode =  event.GetItem()
      self.PopupMenu(HdfTreePopupMenu((self.wxTree,wxNode)), event.GetPoint())

if __name__ == '__main__':
  def GetArgs():
    import sys,argparse #since python 2.7
    exampleCmd='/scratch/detectorData/e14472_00033.hdf5'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__,
                                     epilog='Example:\n'+os.path.basename(sys.argv[0])+' '+exampleCmd+'\n ')
    parser.add_argument('hdfFile',   nargs='?', help='the hdf5 to show')

    args = parser.parse_args()
    return args

  class MyApp(wx.App):

    def OnInit(self):
      args=GetArgs()
      frame = HdfViewerFrame(None, 'h5pyViewer')
      if args.hdfFile:
        frame.OpenFile(args.hdfFile)
      frame.Show(True)
      self.SetTopWindow(frame)
      return True

#------------------ Main Code ----------------------------------
  #redirect stdout/stderr:
  #http://www.blog.pythonlibrary.org/2009/01/01/wxpython-redirecting-stdout-stderr/
  #https://groups.google.com/forum/#!topic/wxpython-users/S9uSKIYdYoo
  #https://17677433047266577941.googlegroups.com/attach/e4d343dc6a751906/REDIRECT.PY?part=2&view=1&vt=ANaJVrFeyCjCMydKnkyfFbYJM7ip07mE-ozUIBxJ5A1QuK1GhycJYJsPTxpAaNk5L2LpXvGhzRPInxDt8_WUcUyK2Ois28Dq8LNebfYoWG9Yxr-tujf5Jk4
  #http://www.wxpython.org/docs/api/wx.PyOnDemandOutputWindow-class.html
  rd=not sys.stdout.isatty()#have a redirect window, if there is no console
  #rd=True #force to open a redirect window
  #rd=False #avoid a redirect window
  app = MyApp(redirect=rd)
  app.MainLoop()
