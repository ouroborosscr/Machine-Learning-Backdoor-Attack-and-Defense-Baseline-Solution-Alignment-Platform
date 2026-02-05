import { alova } from '../request';

/** get role list */
export function fetchGetRoleList(params?: Api.SystemManage.RoleSearchParams) {
  return alova.Get<Api.SystemManage.RoleList>('/systemManage/getRoleList', { params });
}

/**
 * get all roles
 *
 * these roles are all enabled
 */
export function fetchGetAllRoles() {
  return alova.Get<Api.SystemManage.AllRole[]>('/system-manage/roles');
}

/** get user list */
export function fetchGetUserList(data?: Api.SystemManage.UserSearchParams) {
  return alova.Post<Api.SystemManage.UserList>('/system-manage/users/all/', { data });
}

export type UserModel = Pick<
  Api.SystemManage.User,
  'userName' | 'userGender' | 'nickName' | 'userPhone' | 'userEmail' | 'byUserRoleCodeList' | 'statusType'
>;
/** add user */
export function addUser(data: UserModel) {
  return alova.Post<null>('/systemManage/users', data);
}

/** update user */
export function updateUser(data: UserModel) {
  return alova.Patch<null>('/systemManage/users', data);
}

/** delete user */
export function deleteUser(id: number) {
  return alova.Delete<null>(`/system-manage/users/${id}`);
}

/** batch delete user */
export function batchDeleteUser(ids: number[]) {
  return alova.Delete<null>('/system-manage/users', { ids });
}

/** get menu list */
export function fetchGetMenuList() {
  return alova.Get<Api.SystemManage.MenuList>('/system-manage/menus');
}

/** get all pages */
export function fetchGetAllPages() {
  return alova.Get<{ [key: string]: string }>('/system-manage/menus/pages/');
}

/** get menu tree */
export function fetchGetMenuTree() {
  return alova.Get<Api.SystemManage.MenuTree[]>('/system-manage/menus/tree/');
}
