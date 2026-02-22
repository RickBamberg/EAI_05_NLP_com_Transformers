lista = ['adriano', 'karina', 'carlos', 'elvira']
lista2 = []
for a in lista:
    d = ''
    for b in a:
        c = b.upper()
        d = d + c
    print(d)
    lista2.append(d)
    print(lista2)
input('Pressione <Enter> para continuar...')


        