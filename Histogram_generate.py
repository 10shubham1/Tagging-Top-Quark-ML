import sys, os
import matplotlib.pyplot as plt
import numpy as np

pythia_path = '/home/shubham/Yoo/pythia8303/'
cfg = open(pythia_path+"examples/Makefile.inc")
lib = pythia_path+"lib"
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)
print(lib)
import pythia8



def write_mg_cards(PTRANGE, delta=10, nevents=200000, extra=''):
  
    with open('generate_tt.mg5','w') as f:
        f.write("""
    generate p p > t t~
    output jets_tt{3}
    launch
    madspin=none
    done
    set nevents {2}
    set pt_min_pdg {{ 6: {0} }}
    set pt_max_pdg {{ 6: {1} }}
    decay t > w+ b, w+ > j j
    decay t~ > w- b~, w- > j j
    done    
    """.format(PTRANGE[0]-delta, PTRANGE[1]+delta, nevents, extra))

    with open('generate_qcd.mg5','w') as f:
        f.write("""
    generate p p > j j
    output jets_qcd{3}
    launch
    done
    set nevents {2}
    set ptj {0}
    set ptjmax {1}
    done    
    """.format(PTRANGE[0]-delta, PTRANGE[1]+delta, nevents, extra))

def extend_jet_phi(phi, jet_phi):
 
    if abs(jet_phi + np.pi)<1.: # phi close to -pi
        return phi-2*np.pi*int(abs(phi-np.pi) <1-abs(jet_phi + np.pi))
    elif abs(jet_phi - np.pi)<1.: # phi close to pi
        return phi+2*np.pi*int(abs(-phi-np.pi) < 1-abs(jet_phi - np.pi)) 
    else: 
        return phi

def make_image_leading_jet(leading_jet, leading_jet_constituents):
 
    jet_phi = leading_jet.phi()
    jet_eta = leading_jet.eta()
    # redefine grid to only be Delta R=1 around jet center
    yedges = [phi for phi in phiedges if abs(phi-jet_phi)<=1.2+(phiedges[1]-phiedges[0])]
    xedges = [eta for eta in etaedges if abs(eta-jet_eta)<=1.2+(etaedges[1]-etaedges[0])]
    
    jet_constituents = np.array([ [c.pT(), c.eta(), extend_jet_phi(c.phi(), jet_phi) ] for c in leading_jet_constituents ])

    histo, xedges, yedges =  np.histogram2d(jet_constituents[:,1], jet_constituents[:,2], bins=(xedges,yedges), weights=jet_constituents[:,0])
    
    
    return histo.T, (xedges, yedges)
    
def make_image_event(all_jets, all_constituents):

    out=[]
    for i in range(len(all_jets)):
        jet_phi = all_jets[i].phi()
        
        jet_constituents = np.array([ [c.pT(), c.eta(), extend_jet_phi(c.phi(), jet_phi) ] for c in all_constituents[i] ])

        histo, xedges, yedges =  np.histogram2d(jet_constituents[:,1], jet_constituents[:,2],bins=(etaedges,phiedges),weights=jet_constituents[:,0])
    
        
        out.append(histo.T)
    
    return out, (xedges, yedges)

def run_pythia_get_images(lhe_file_name, PTRANGE = [500., 700.], PTRANGE2=None, nevents=10**6, plot_first_few=True):

    
    if lhe_file_name.endswith('gz') and not os.path.isfile(lhe_file_name.split('.gz')[0]): 
        os.system('gunzip < {} > {}'.format(lhe_file_name, lhe_file_name.split('.gz')[0]))
    lhe_file_name = lhe_file_name.split('.gz')[0]
    if not os.path.isfile(lhe_file_name): raise Exception('no LHE file')
    
    PTRANGE2 = PTRANGE if PTRANGE2 is None else PTRANGE2
    
    pythia = pythia8.Pythia()
    
    pythia.readString("Beams:frameType = 4")
    pythia.readString("Beams:LHEF = "+lhe_file_name)

    pythia.init()
    # jet parameters: antikt, R, pT_min, Eta_max
    slowJet = pythia8.SlowJet(-1, 1.0, 20, 2.5)

    
    leading_jet_images, all_jet_images = [], []
    jetpep=[]
    jetpep1=[]
    iplot=0
    
    for iEvent in range(0,nevents):
        if not pythia.next(): continue

        #print('{}\r'.format(iEvent//10*10)),

        
        slowJet.analyze(pythia.event)

        njets = len([j for j in range(0,slowJet.sizeJet()) if slowJet.p(j).pT()> PTCUT])

        jetpep1.append([[slowJet.p(j).pT(), slowJet.p(j).eta(), slowJet.p(j).phi()] for j in range(0, njets)])
        jet_list = [ slowJet.p(j) for j in range(0, njets)] 
        jet_constituent_list=[ [ pythia.event[c].p() for c in slowJet.constituents(j)] for j in range(0, njets)]

        # at least two high-pT large R jets in the right range
        if njets<2: continue 
        if not (PTRANGE[0] < jetpep1[iEvent][0][0] < PTRANGE[1] and PTRANGE2[0] < jetpep1[iEvent][1][0] < PTRANGE2[1]): continue

        hh, (xx, yy) = make_image_leading_jet(jet_list[0], jet_constituent_list[0])
        hh1, _ = make_image_event(jet_list, jet_constituent_list)
        
        jetpep.append([[slowJet.p(j).pT(), slowJet.p(j).eta(), slowJet.p(j).phi()] for j in range(0, njets)])
        leading_jet_images.append(hh)
        all_jet_images.append(hh1)
        
    return leading_jet_images, all_jet_images, np.array(jetpep)


def pad_image(image):

    max_size=[16,22]
    size = list(np.shape(image))
    #print(size)
    px, py = max_size[0]-size[0], max_size[1]-size[1]
    i=np.pad(image, (  (  int(np.floor(px/2)) , int(np.ceil(px/2))  )   , ( int(np.floor(py/2)) ,int(np.ceil(py/2)) )   ), 'constant')
    return i
    
def normalize(histo, multi=255):
    
    return (histo/np.max(histo)*multi).astype(int)
    
    

etaedges = np.arange(-3,3+0.01,0.12)
phiedges = np.arange(-np.pi*4/3,np.pi*4/3+0.01,np.pi/18.)
cmap = plt.get_cmap('gray_r')
PTCUT = 50.
nevents = 50000


if __name__ == "__main__":

    outdir = 'images_out/'
    if not os.path.isdir(outdir): os.system('mkdir {}'.format(outdir))
    cwd = os.getcwd()


    write_mg_cards([500,700], nevents=nevents)

    #os.system('cd ../../MG5_aMC_v2_9_2 ;ls; bin/mg5_aMC  {}'.format(os.path.join(cwd, 'generate_tt.mg5')))
    #os.system('cd ../../MG5_aMC_v2_9_2 ; bin/mg5_aMC  {}'.format(os.path.join(cwd, 'generate_qcd.mg5')))

    #lhe_file_name = '/home/shubham/Yoo/MG5_aMC_v2_9_2/jets_tt/Events/run_01_decayed_1/unweighted_events.lhe.gz'
    #leading_jet_images, all_jet_images, jetpep = run_pythia_get_images(lhe_file_name, PTRANGE=[500,700], PTRANGE2=[450,700])

    #np.savez_compressed(outdir+'tt_leading_jet.npz', leading_jet_images)
    #np.savez_compressed(outdir+'tt_all_jets.npz', all_jet_images)
    #np.savez_compressed(outdir+'tt_jetpep.npz', jetpep)

    lhe_file_name = '/home/shubham/Yoo/MG5_aMC_v2_9_2/jets_qcd/Events/run_01/unweighted_events.lhe.gz'
    leading_jet_images1, all_jet_images1, jetpep1 = run_pythia_get_images(lhe_file_name, PTRANGE=[500,700], PTRANGE2=[450,700])

    np.savez_compressed(outdir+'qcd_leading_jet.npz', leading_jet_images1)
    np.savez_compressed(outdir+'qcd_all_jets.npz', all_jet_images1)
    np.savez_compressed(outdir+'qcd_jetpep.npz', jetpep1)

    