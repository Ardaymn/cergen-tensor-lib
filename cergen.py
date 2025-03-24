import random

def cekirdek(sayi: int):
    random.seed(sayi)

def rastgele_dogal(boyut, aralik=(0,100), dagilim='uniform'):
    if dagilim != 'uniform':
        raise ValueError("dagilim parameter is given differently")

    def generate_gergen(dimensions):
        if not dimensions:
            return None
        if len(dimensions) == 1:
            return [random.randint(*aralik) for _ in range(dimensions[0])]
        else:
            return [generate_gergen(dimensions[1:]) for _ in range(dimensions[0])]

    return gergen(generate_gergen(boyut))

def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    if dagilim != 'uniform':
        raise ValueError("dagilim parameter is given differently")

    def generate_gergen(dimensions):
        if not dimensions:
            return None
        if len(dimensions) == 1:
            return [random.uniform(*aralik) for _ in range(dimensions[0])]
        else:
            return [generate_gergen(dimensions[1:]) for _ in range(dimensions[0])]

    return gergen(generate_gergen(boyut))





class Operation:
    def __call__(self, *operands):
        """
        Makes an instance of the Operation class callable.
        Stores operands and initializes outputs to None.
        Invokes the forward pass of the operation with given operands.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the forward pass of the operation.
        """
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError



import math
from typing import Union

class gergen:

    __veri = None #A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None #Dimensions of the derivative (Shape)


    def __init__(self, veri=None):
        if veri is None:
            self.__veri = veri
            self.D = None
            self.__boyut = None
        elif isinstance(veri, (int, float)):
            self.__veri = veri
            self.D = veri
            self.__boyut = ()
        else:
            self.__veri = veri
            
            self.__boyut = self.boyut_hesaplama_yardımcı()
            self.D = self.transpose_helper(self.__veri)

    def boyut_hesaplama_yardımcı(self):

        boyut = []
        eleman = self.__veri
        while isinstance(eleman, list):
            boyut.append(len(eleman))
            eleman = eleman[0]
        return tuple(boyut)

    def transpose_helper(self, veri):
        if not isinstance(veri, list) or not veri or not isinstance(veri[0], list):
            return veri  # Skaler veya 1D listeyi aynı bırak.
        return list(map(list, zip(*veri)))



    def get_item_yardımcı(veri, index):
        if len(index) == 1:
            return veri[index[0]]
        else:
            return gergen.get_item_yardımcı(veri[index[0]], index[1:])

    def __getitem__(self, index):
        if isinstance(index, (int, float)):
            return self.__veri[index]
        elif isinstance(index, tuple):
            return self.get_item_yardımcı(self.__veri, index)




    def _str_yardimci(self, veri, derinlik):
        if not isinstance(veri, list) or not veri or not isinstance(veri[0], list):
            return str(veri) #buna tekrar bak

        ic_sonuc = "["
        for i, alt_veri in enumerate(veri):
            sonlandirma = ",\n" + " " * (derinlik + 1) if i < len(veri) - 1 else ""
            ic_sonuc += self._str_yardimci(alt_veri, derinlik + 1) + sonlandirma
        ic_sonuc += "]"
        return ic_sonuc

    def __str__(self):
        if self.__veri is None:
            return "0 boyutlu gergen:" #buna da tekrar bak
        if isinstance(self.__veri,int) or isinstance(self.__veri,float):
            return "0 boyutlu skaler gergen:\n" + str(self.__veri)

        boyut_str = 'x'.join(map(str, self.__boyut)) + " boyutlu gergen:\n"
        return boyut_str + self._str_yardimci(self.__veri, 0)




    def __mul__(self, other):

        if isinstance(other, (int, float)):
            def mul_scalar(data, scalar):
                if isinstance(data, list):
                    return [mul_scalar(sub, scalar) for sub in data]
                else:
                    return data * scalar
            new_data = mul_scalar(self.__veri, other)
            return gergen(new_data)


        elif isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError()

            def mul_elementwise(data1, data2):
                if isinstance(data1, list) and isinstance(data2, list):
                    return [mul_elementwise(sub1, sub2) for sub1, sub2 in zip(data1, data2)]
                else:
                    return data1 * data2
            new_data = mul_elementwise(self.__veri, other.__veri)
            return gergen(new_data)

        else:
            raise TypeError()


    def __truediv__(self, other):

        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError()
            return gergen(self.scalar_divide(self.__veri, other))


        elif isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError()
            return gergen(self.elementwise_divide(self.__veri, other.__veri))

        else:
            raise TypeError()

    

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        # Handle division of a scalar by a gergen instance.
        if isinstance(other, (int, float)):
            return gergen(self.scalar_rdivide(other, self.__veri))
        else:
            raise TypeError("Unsupported type for rtruediv")

    def scalar_rdivide(self, scalar, data):
        
        if isinstance(data, list):
            return [self.scalar_rdivide(scalar, sub) for sub in data]
        else:
            return scalar / data

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        
        if isinstance(other, (int, float)):
            return gergen(self.scalar_rsubtract(other, self.__veri))
        else:
            raise TypeError("Unsupported type for rsub")

    def scalar_rsubtract(self, scalar, data):
        # Recursively subtract the data in gergen from the scalar.
        if isinstance(data, list):
            return [self.scalar_rsubtract(scalar, sub) for sub in data]
        else:
            return scalar - data

    


    def scalar_divide(self, data, scalar):

        if isinstance(data, list):
            return [self.scalar_divide(sub, scalar) for sub in data]
        else:
            return data / scalar

    def elementwise_divide(self, data1, data2):

        if isinstance(data1, list) and isinstance(data2, list):
            return [self.elementwise_divide(sub1, sub2) for sub1, sub2 in zip(data1, data2)]
        else:
            if data2 == 0:
                raise ZeroDivisionError()
            return data1 / data2



    def add_scalar(self, data, scalar):
        if isinstance(data, list):
            return [self.add_scalar(sub, scalar) for sub in data]
        else:
            return data + scalar

    def add_elementwise(self, data1, data2):
        if isinstance(data1, list) and isinstance(data2, list):
            return [self.add_elementwise(sub1, sub2) for sub1, sub2 in zip(data1, data2)]
        else:
            return data1 + data2

    def __add__(self, other):
        if isinstance(other, (int, float)):

            new_data = self.add_scalar(self.__veri, other)
            return gergen(new_data)
        elif isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError("Dimensions do not match.")

            new_data = self.add_elementwise(self.__veri, other.__veri)
            return gergen(new_data)
        else:
            raise TypeError("Unsupported type for addition.")





    def __sub__(self, other):

        if isinstance(other, (int, float)):
            return gergen(self.scalar_subtract(self.__veri, other))


        elif isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError()
            return gergen(self.elementwise_subtract(self.__veri, other.__veri))

        else:
            raise TypeError()

    def scalar_subtract(self, data, scalar):

        if isinstance(data, list):
            return [self.scalar_subtract(sub, scalar) for sub in data]
        else:
            return data - scalar

    def elementwise_subtract(self, data1, data2):

        if isinstance(data1, list) and isinstance(data2, list):
            return [self.elementwise_subtract(sub1, sub2) for sub1, sub2 in zip(data1, data2)]
        else:
            return data1 - data2




    def uzunluk_helper(self, veri):

        if not isinstance(veri, list):
            return 1

        if len(veri) == 0:
            return 0

        return sum(self.uzunluk_helper(item) for item in veri)

    def uzunluk(self):

        if self.__veri is None:
            return 0

        return self.uzunluk_helper(self.__veri)

    def boyut(self):
        return self.__boyut

    def devrik(self):
        transposed_data = self.transpose_helper(self.__veri)
        return gergen(transposed_data)

    def sin(self):

        def sin_rekürsif(veri):
            if veri is None: #!!!!
                return None

            if isinstance(veri, (int, float)):
                return math.sin(veri)

            elif isinstance(veri, list):
                return [sin_rekürsif(eleman) for eleman in veri]


        return gergen(sin_rekürsif(self.__veri))

    def cos(self):
        def cos_rekürsif(veri):
                if veri is None: #!!!!
                    return None

                if isinstance(veri, (int, float)):
                    return math.cos(veri)

                elif isinstance(veri, list):
                    return [cos_rekürsif(eleman) for eleman in veri]


        return gergen(cos_rekürsif(self.__veri))

    def tan(self):
        def tan_rekürsif(veri):
            if veri is None: #!!!!
                return None

            if isinstance(veri, (int, float)):
                return math.tan(veri)

            elif isinstance(veri, list):
                return [tan_rekürsif(eleman) for eleman in veri]


        return gergen(tan_rekürsif(self.__veri))

    def us(self, n: int):

        def us_alma(sayi, n):
            return sayi ** n

        def rekürsif(veri):
            if veri is None: #!!!!
                return None

            if isinstance(veri, (int, float)):
                return us_alma(veri,n)

            elif isinstance(veri, list):
                return [rekürsif(eleman) for eleman in veri]


        return gergen(rekürsif(self.__veri))

    def log(self):
        def rekürsif(veri):
            if veri is None: #!!!!
                return None

            if isinstance(veri, (int, float)):
                return math.log10(veri)

            elif isinstance(veri, list):
                return [rekürsif(eleman) for eleman in veri]


        return gergen(rekürsif(self.__veri))

    def ln(self):
        def rekürsif(veri):
            if veri is None: #!!!!
                return None

            if isinstance(veri, (int, float)):
                return math.log(veri)

            elif isinstance(veri, list):
                return [rekürsif(eleman) for eleman in veri]


        return gergen(rekürsif(self.__veri))

    def L1_yardimci(self,veri):
        if not isinstance(veri, list):
            return abs(veri)

        if len(veri) == 0:
            return 0

        elif isinstance(veri, list):
            return sum(self.L1_yardimci(item) for item in veri)

    def L1(self):
        if self.__veri is None:
            return 0

        return self.L1_yardimci(self.__veri)

    def L2_yardimci(self, veri):

        if isinstance(veri, (int, float)):
            return veri ** 2

        elif isinstance(veri, list):
            return sum(self.L2_yardimci(item) for item in veri)

        else:
            return 0

    def L2(self):

        if self.__veri is None:
            return 0

        return math.sqrt(self.L2_yardimci(self.__veri))


    def Lp_yardimci(self, veri, p):

        if isinstance(veri, (int, float)):
            return math.pow(abs(veri), p)

        elif isinstance(veri, list):
            return sum(self.Lp_yardimci(item, p) for item in veri)

        else:
            return 0

    def Lp(self, p: int):

        if not isinstance(p, int) or p <= 0:
            raise ValueError()
        return math.pow(self.Lp_yardimci(self.__veri, p), 1/p)

    def listeye(self):

        if isinstance(self.__veri, list):
            return self.__veri
        elif isinstance(self.__veri, (int, float)):
            return [self.__veri]
        else:
            return []

    def duzlestir(self):
        def flatten(data):
            if isinstance(data, list):
                return [element for sublist in data for element in flatten(sublist)]
            else:
                return [data]

        flattened_data = flatten(self.__veri)
        return gergen(flattened_data)

    # def boyutlandır_yardımcı(self,yeni_boyut):



    def boyutlandir(self, yeni_boyut):
        if yeni_boyut is None:
            raise ValueError()

        result = self.duzlestir()
        yeni_uzunluk = result.uzunluk()

        eleman_sayisi = 1
        for boyut in yeni_boyut:
            eleman_sayisi *= boyut

        if eleman_sayisi != yeni_uzunluk:
            raise ValueError()



   

    
    def ic_carpim(self, other):
        
        if not isinstance(other, gergen):
            raise TypeError()

        
        if not (1 <= len(self.__boyut) <= 2 and 1 <= len(other.__boyut) <= 2):
            raise ValueError()

        
        if len(self.__boyut) == len(other.__boyut) == 1:
            if self.__boyut != other.__boyut:
                raise ValueError()
            return sum(x * y for x, y in zip(self.__veri, other.__veri))

        
        elif len(self.__boyut) == len(other.__boyut) == 2:
            if self.__boyut[1] != other.__boyut[0]:
                raise ValueError()
            result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*other.__veri)] for row in self.__veri]
            return gergen(result)

        else:
            raise ValueError()


   


    def dis_carpim(self, other):

        if not isinstance(other, gergen):
            raise TypeError()
        
        if len(self.__boyut) != 1 or len(other.__boyut) != 1:
            raise ValueError()
        
        
        result = [[x * y for y in other.__veri] for x in self.__veri]
        return gergen(result)

    def topla_recursive(self, veri, eksen=None, derinlik=0):
        if not isinstance(veri, list):
            return veri
        
        if eksen is None:
            return sum(self.topla_recursive(sub, eksen, derinlik + 1) for sub in veri)

        elif derinlik == eksen:
            return [self.topla_recursive(sub, eksen, derinlik + 1) for sub in veri]
        
        else:
            return sum(self.topla_recursive(sub, eksen, derinlik + 1) for sub in veri)

    def topla(self, eksen=None):
        if eksen is None:
            return self.topla_recursive(self.__veri)
        
        toplam = self.topla_recursive(self.__veri, eksen)

        if isinstance(toplam, list):
            return gergen(toplam)
        else:
            return gergen([toplam])



    def ortalama(self, eksen=None):
        toplam = self.topla(eksen=eksen)
        if eksen is None:
            
            uzunluk = self.uzunluk()
            return gergen(self.scalar_divide(toplam.listeye(), uzunluk))
        else:
            
            uzunluk = self.__boyut[eksen]
            return gergen(self.scalar_divide(toplam.listeye(), uzunluk))


def example_1():
    #Example 1
    boyut = (64,64)
    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)

    start = time.time()
    #TODO
    #Apply given equation
    result = g1.ic_carpim(g2)
    end = time.time()

    start_np = time.time()
    #Apply the same equation for NumPy equivalent
    end_np = time.time()


    np_g1 = np.array(g1.listeye())
    np_g2 = np.array(g2.listeye())

    start_np = time.time()

    result2 = np.dot(np_g1, np_g2)

    end_np = time.time()

    result1 = np.array(result1.listeye())
    #TODO:
    #Compare if the two results are the same
    #Report the time difference
    print("Time taken for gergen:", end-start)
    print("Time taken for numpy:", end_np-start_np)
    print("Result", result1- result2)

def example_2():
    boyut = (4,16,16,16)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)
    c = rastgele_gercek(boyut)

    start = time.time()

    result1 = (a*b + a*c + b*c).ortalama()

    end = time.time()

    np_1 = np.array(a.listeye())
    np_2 = np.array(b.listeye())
    np_3 = np.array(c.listeye())

    start_np = time.time()

    result2 =(np_1*np_2 + np_1*np_3 + np_2*np_3).mean()

    end_np = time.time()

    print("Time taken for gergen:", end-start)
    print("Time taken for numpy:", end_np-start_np)
    print("Difference:", result1- result2)

def example_3():
    boyut = (3,64,64)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)

    start = time.time()

    result1 = ((a.sin() + b.cos()).ln()).us(2) / 8

    end = time.time()

    np_a = np.array(a.listeye())
    np_b = np.array(b.listeye())

    start_np = time.time()

    result2 = (np.log((np.sin(np_a) + np.cos(np_b))))**2 / 8

    end_np = time.time()

    result1 = np.array(result1.listeye())

    print("Time taken for gergen:", end-start)
    print("Time taken for numpy:", end_np-start_np)
    print("Difference:", result1- result2)
