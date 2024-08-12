from flask import Flask,render_template,request
import pickle
application=Flask(__name__)
app=application

ridge_modle=pickle.load(open(r'models\algerian_ridge_model.pkl','rb'))
scaler=pickle.load(open(r'models\algerian_ridge_scaler.pkl','rb'))
@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        temperature=request.form.get('temperature')
        rh=request.form.get('rh')
        Ws=request.form.get('Ws')
        rain=request.form.get('rain')
        FFMC=request.form.get('FFMC')
        DMC=request.form.get('DMC')
        ISI=request.form.get('ISI')
        classes=request.form.get('classes')
        region=request.form.get('region')
        
        ##data preparation
        data=[[temperature,rh, Ws, rain,FFMC, DMC,ISI,classes,region]]
        #scaling of the data
        scaled_data=scaler.transform(data)
        
        #predicstion for the given data
        fwi=ridge_modle.predict(scaled_data)
        print(f'############### {fwi} #################')
        return render_template('index.htm',fwi=fwi)
    return render_template('index.htm')

if __name__=='__main__':
    app.run(debug=True)