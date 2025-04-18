class DaMaSCUS_helper:
    def __init__(self,samplesize=None,isorings=None):
        if samplesize is not None:
            self.samplesize= samplesize
        else:
            self.samplesize=1000000

        if isorings is not None:
            self.isorings = isorings
        else:
            self.isorings=180

        self.me_eV = 5.1099894e5
        self.mP_eV = 938.27208816 *1e6
        self.km = 5.067*1e18
        self.s = 1.51905*1e24
        self.rhoX = 0.3
        self.QEDark_outdir = '../sensei_toy_limit/python/theory_tools/QEDark/halo_data/modulated/'

    def update_samplesize(self,sample_size):
        self.samplesize=sample_size
    def update_isorings(self,isorings):
        self.isorings=isorings
    def mu_Xe(self,mX):
        """
        DM-electron reduced mass
        """
        return mX*self.me_eV/(mX+self.me_eV)

    def mu_XP(self,mX):
        """
        DM-proton reduced mass
        """
        return mX*self.mP_eV/(mX+self.mP_eV)

    def sigmaE_to_sigmaP(self,sigmaE,mX):
        import numpy as np
        mX*=1e6 #eV
        sigmaP = sigmaE*(self.mu_XP(mX)/self.mu_Xe(mX))**2
        # sigmaP = np.round(sigmaP,3)
        return sigmaP
    
    def sigmaP_to_sigmaE(self,sigmaP,mX):
        import numpy as np
        mX*=1e6 #eV
        sigmaE = sigmaP*(self.mu_Xe(mX)/self.mu_XP(mX))**2
        # sigmaP = np.round(sigmaP,3)
        return sigmaE
    

    def make_DaMaSCUS(self,clean = False):
        import os
        if clean:
            success = os.system('(make clean)')
        else:
            success = os.system('(make)')

        if success != 0:
            print('Something went wrong. Make sure you are able to make DaMaSCUS on your machine.\n If not, edit the makefile with whatever configurations your system needs')
        return success

    def write_cfg(self,mass,sigmaP,fdm):
        ##takes sigma in cm^2##
        if fdm == 0:
            form_factor = "ChargeScreening"
        elif fdm == 2:
            form_factor = "LightMediator"

        else:
            form_factor = "None"
        fname = f'mX{mass}_sigma{sigmaP}_fdm{fdm}.cfg'
        simid =f"sensei_mX{mass}_sigma{sigmaP}_fdm{fdm}"
        with open(f'./bin/{fname}','w') as f:
            f.write('//DaMaSCUS Configuration File\n\n')
            f.write('//Simulation input parameter\n')
            f.write(
                f'simID		=	"{simid}";	//MC Simulation ID\n \
                initialruns	=	10000000L;		//Number of particles in the initial MC run\n\
                samplesize	=	{self.samplesize};			//velocity sample size per isodetection ring\n\
                vcutoff		=	1.0;			//velocity cutoff in cm/sec\n\
                rings		=	{self.isorings};				//number of isodetection rings\n\n\n')

            f.write('//Simulation Time:\n \
                date		=	[02,02,2022];	//Date [dd,mm,yyyy]\n \
                time		=	[0,0,0];		//Universal time [h,m,s]\n\n\n')
            

            f.write(f'//Dark Matter Data\n\
            //Particle data\n\
                mass		=	{mass};			//in MeV\n\
                sigma 		=	{sigmaP};			//in cm^2 \n\
                formfactor	=	"{form_factor}";			//Options: "None", "HelmApproximation"\n\n\
        \
            //DM Halo \n\
                halomodel	=	"SHM";			//Options: Standard Halo Model "SHM",...\n\
                rho			=	{self.rhoX};			//DM halo energy density in GeV/cm^3\n\
        \n\
        //Detector depth: \n\
                depth		=	1400.0;			//in meter \n\
        \
        //Analysis parameter \n\
                experiment	=	"None";			//Options: "LUX" for heavy DM,"CRESST-II" for light DM and "None"')
        return fname,simid
            
    def run_simulation(self,mass,sigmaE,fdm,ncores=2,overwrite=False):
        import os
        import numpy as np
        if fdm == 0:
            dirname = 'Parameter_Scan_Scr'
        elif fdm == 2:
            dirname = 'Parameter_Scan_LM'
        if ncores == 'all':
            arg = '--use-hwthread-cpus'
        else:
            arg = f'-n {ncores}'
        mass_str = str(np.round(mass,3)).replace('.','_')
        mass = np.round(mass,3)
        write_dir = self.QEDark_outdir + dirname + f'/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'
        if os.path.isdir(write_dir) and len(os.listdir(write_dir)) > 30 and not overwrite:
            print(f"this point already generated with {len(os.listdir(write_dir))} isoangles")
            print(f'sigmaE: {sigmaE}. Mass: {mass}')
            return 
        sigmaP = self.sigmaE_to_sigmaP(sigmaE,mass)
        sigmaP = float(format(sigmaP,'0.2e'))
        fname,idname = self.write_cfg(mass,sigmaP,fdm)
        success = os.system(f'(cd bin && mpirun {arg} ./DaMaSCUS-Simulator {fname})')
        if success != 0:
            print('Something went wrong with simulations')
        else:
            success2 = os.system(f'(cd bin && mpirun {arg} ./DaMaSCUS-Analyzer {idname})')
            if success2 != 0:
                print('Something went wrong with analysis')
                
            else:
                self.fix_eta(mass,sigmaE,fdm,idname,self.QEDark_outdir)

    def fix_eta(self,mass,sigmaE,fdm,idname,outdir,delete=True,overwrite=True):
        from scipy.interpolate import PchipInterpolator
        import numpy as np
        import os
        # os.system('rm -r ./data/*')
        fname_rhoDam = f'./results/{idname}.rho'
        rhofiledata = np.loadtxt(fname_rhoDam,delimiter='\t')
            
        rho = rhofiledata[:,1]
        rho_i = rhofiledata[:,0]
        rho_func = PchipInterpolator(rho_i,rho)

        mass_str = str(np.round(mass,3)).replace('.','_')
        if fdm == 0:
            dirname = 'Parameter_Scan_Scr'
        elif fdm == 2:
            dirname = 'Parameter_Scan_LM'

        write_dir = outdir + dirname + f'/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
        else:
            if overwrite:
                 os.system(f"rm -r {write_dir}/*.txt")
        num_rings = len(os.listdir(f'./results/{idname}_histograms/')) //2
        actual_angles = np.linspace(0,180,num_rings)
        for isoangle in range(num_rings):
            fname_DAMASCUS = f'./results/{idname}_histograms/eta.{isoangle}'
            ai = actual_angles[isoangle]
            fdata = np.loadtxt(fname_DAMASCUS,delimiter='\t')
            vmin = fdata[:,0] * self.s/self.km
            eta = fdata[:,1]*self.km/self.s
            eta_err = fdata[:,3]*self.km/self.s

            #filter vmin = 0
            filter_indices = np.where(vmin > 0)
            vmin = vmin[filter_indices]
            eta = eta[filter_indices]
            eta_err = eta_err[filter_indices]


            eta*=(rho_func(ai)/self.rhoX)
            # print(rho[isoangle]/self.rhoX,self.rhoX,rho[isoangle])
            eta_err*=(rho_func(ai)/self.rhoX)
            import csv
            with open(write_dir+f'DM_Eta_theta_{isoangle}.txt','w') as f:
                writer = csv.writer(f,delimiter='\t')
                writer.writerows(zip(vmin,eta,eta_err))
        if delete:
            os.system(f"rm -r ./results/{idname}*")
        return




class Earth_Density_Layer_NU:
    def __init__(self):
        self.Elements = None
        self.GeV = 1.0
        self.MeV	 = 1.0E-3 * self.GeV
        self.eV	 = 1.0E-9 * self.GeV
        self.gram = 5.617977528089887E23 * self.GeV
        self.cm			 = 5.067E13 / self.GeV
        self.meter		 = 100 * self.cm
        self.km			 = 1000 * self.meter
        self.EarthRadius = 6371 *self.km
        self.mNucleon  = 0.932 * self.GeV
        self.mProton  = 0.938 * self.GeV
        self.m  = 0.932 * self.GeV
        self.sec = 299792458 * self.meter
        self.Bohr_Radius = 5.291772083e-11 * self.meter
        self.mElectron = 0.511 * self.MeV
        self.alpha= 1.0 / 137.035999139


    def get_layer(self,r): #inner core
        #r in natural units
        x = r / self.EarthRadius
        # print('converted radius',r/self.km)
        if r < 1221.5*self.km: #km
            # print('Inner Core')
            self.Core()
            self.density = 13.0885 - 8.8381*x**2
        elif r >= 1221.5*self.km and r < 3480*self.km: #outer core
            # print('Outer Core')
            self.Core()
            self.density = 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
        elif r >= 3480*self.km and r < 3630*self.km: #Lower Mantle 1 
            # print('Lower Mantle 1')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3
        elif r >= 3630*self.km and r < 5600*self.km: #Lower Mantle 2
            # print('Lower Mantle 2')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3
        elif r >= 5600*self.km and r < 5701*self.km: #Lower Mantle 3
            # print('Lower Mantle 3')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3

        elif r >= 5701*self.km and r < 5771*self.km:#Transition Zone 1
            # print('Transition Zone 1')

            self.Mantle()
            self.density = 5.3197 - 1.4836*x
        elif r >= 5771*self.km and r < 5971*self.km:#Transition Zone 2
            # print('Transition Zone 2')
            self.Mantle()
            self.density = 11.2494 - 8.0298*x
        elif r >= 5971*self.km and r < 6151*self.km: #Transition Zone 3
            # print('Transition Zone 3')

            self.Mantle()
            self.density = 7.1089-3.8405*x
        elif r >= 6151*self.km and r < 6291*self.km: #LVZ
            # print('LVZ')

            self.Mantle()
            self.density = 2.6910 + 0.6924*x
        elif r >= 6291*self.km and r < 6346.6*self.km: #LID
            # print('LID')
            self.Mantle()
            self.density = 2.6910 + 0.6924*x
        elif r >= 6346.6*self.km and r < 6356*self.km: #crust 1 
            # print('Inner Crust')
            self.Mantle()
            self.density = 2.9
        elif r >= 6356*self.km and r < 6368*self.km: #crust 2
            # print('Outer Crust')
            self.Mantle()
            self.density = 2.6
        # elif r >= 6368 and r < 6371: #ocean
        #     self.Mantle()
        self.density*= self.gram * (self.cm)**(-3) #[GeV^4]
        return

        
        

    def Core(self):
        self.Elements = [
            [26, 56, 0.855],  # # Iron			Fe
            [14, 28, 0.06],	   ## Silicon		Si
            [28, 58, 0.052],   ## Nickel		Ni
            [16, 32, 0.019],   ## Sulfur		S
            [24, 52, 0.009],   ## Chromium		Cr
            [25, 55, 0.003],   ## Manganese    Mn
            [15, 31, 0.002],   ## Phosphorus	P
            [6, 12, 0.002],	   ## Carbon		C
            [1, 1, 0.0006]	   ## Hydrogen		H
        ]
        return

    def Mantle(self):
        self.Elements = [
            [8, 16, 0.440],		# Oxygen		O
		[12, 24, 0.228],	# Magnesium		Mg
		[14, 28, 0.21],		# Silicon		Si
		[26, 56, 0.0626],	# Iron			Fe
		[20, 40, 0.0253],	# Calcium		Ca
		[13, 27, 0.0235],	# Aluminium		Al
		[11, 23, 0.0027],	# Natrium		Na
		[24, 52, 0.0026],	# Chromium		Cr
		[28, 58, 0.002],	# Nickel		Ni
		[25, 55, 0.001],	# Manganese		Mn
		[16, 32, 0.0003],	# Sulfur		S
		[6, 12, 0.0001],	# Carbon		C
		[1, 1, 0.0001],		# Hydrogen		H
		[15, 31, 0.00009]	# Phosphorus	P
        ]
        return 
    
    def NucleusMass(self,N):
        return N*self.mNucleon


 
    


    def muXElem(self,mX,mElem):     
      return mX*mElem/(mX+mElem)



    def sigma_i(self,v,isotope_mass,z,sigmaP,mX,FDMn,doScreen=True):
        import numpy as np
        qmax = 2 *  self.muXElem(mX,isotope_mass) * v
        q2max = qmax*qmax
        qref = self.alpha * self.mElectron
        # sigmaP_bar = 18 * pi * alpha*alphaD * epsilon^2 * muXP ^2 / (qref + ma_prime^2)^2
        # a = 1 /4 (9 pi^2 / 2*Z) ^ 1/3
        # a0 = 0.89 *a0 / z^1/3
        a = (1/4)*((9*np.pi**2)/2/z)**(1/3)*self.Bohr_Radius
        
        x = a*a*q2max
        y = a*a * qref*qref
        
        if FDMn == 1:
            fdm_factor = 1
        elif FDMn == 2:
            doScreen = False
            fdm_factor = y*y / (1+x)
        if doScreen:
            fdm_factor = (1+ (1/(1+x)) - (2/x)*np.log(1+x))
        si= sigmaP*((self.muXElem(mX,isotope_mass)/self.muXElem(mX,self.mNucleon))**2) *(z**2)* fdm_factor 

        return si #same units as sigmaP



    def Mean_Free_Path(self,r,mX,sigmaP,v,FDMn,doScreen=True):
        #r in natural units
        #mX in GeV
        #v in c
        #sigmaP in cm^2
        #convert sigmaP into energy units
        

        lambda_inv = 0
        sigmaP *= self.cm**2# [1/ev^2]
        self.get_layer(r) 
        density = self.density #natural units
        num_isotopes = len(self.Elements)
        for i in range(num_isotopes):
            Element= self.Elements[i]
            fractional_density = Element[2]
            Z = Element[0]
            N = Element[1]
            isotope_mass = self.NucleusMass(N) #GeV
            si = self.sigma_i(v,isotope_mass,Z,sigmaP,mX,FDMn,doScreen)
            # print('fractional_density,density/isotope_mass,si')
            # print(fractional_density,density/isotope_mass,si)
# 
            lambda_inv+= fractional_density * (density/isotope_mass) *si #in units of GeV
        
        mfp = 1/lambda_inv #[1/GeV]
        mfp /= self.EarthRadius
        return mfp
    

